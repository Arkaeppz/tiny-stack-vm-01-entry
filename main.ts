// A tiny stack VM for make.fun's tiny stack ISA v1, following ISA.md rule by rule,
// including the ones most implementations get wrong: exact wrapping multiplication,
// unsigned division and remainder, every trap with its own exit code, and the stack,
// call, memory, step and output limits, each checked in the order the ISA states.
// For every instance under the input directory it writes the program's output bytes
// to stdout.bin and its exit code (halt, EXIT, or a trap's code) to exit.txt.
import * as fs from 'node:fs';
import * as path from 'node:path';

// Trap exit codes: 2 underflow, 3 overflow, 4 division by zero, 5 unknown opcode,
// 6 memory address out of range, 7 call stack full, 8 RET with no caller, 9 pc or
// immediate outside the program, 10 step limit, 11 output limit.

// [pops, pushes] for the opcodes that are not binary operations; binary ones pop 2, push 1.
const PP: Record<number, [number, number]> = {
  0x01: [0, 1], 0x02: [0, 1], 0x03: [1, 0], 0x04: [1, 2], 0x05: [2, 2], 0x06: [2, 3],
  0x07: [3, 3], 0x17: [1, 1], 0x1b: [1, 1], 0x31: [1, 0], 0x32: [1, 0], 0x40: [1, 1],
  0x41: [2, 0], 0x50: [1, 0], 0x51: [1, 0], 0x52: [0, 1], 0x53: [1, 0],
};
const BIN = (op: number): boolean => op >= 0x10 && op <= 0x22 && op !== 0x17 && op !== 0x1b;
const KNOWN = (op: number): boolean =>
  op <= 0x07 || (op >= 0x10 && op <= 0x22) || (op >= 0x30 && op <= 0x34) || op === 0x40 || op === 0x41 || (op >= 0x50 && op <= 0x53);
const IMM = (op: number): number => (op === 0x01 ? 1 : op === 0x02 ? 4 : op >= 0x30 && op <= 0x33 ? 2 : 0);

function run(p: Uint8Array, inp: Uint8Array): { out: Uint8Array; code: number } {
  const n = p.length;
  const D = new Int32Array(256);
  const C = new Int32Array(64);
  const M = new Int32Array(256);
  const out = new Uint8Array(65536);
  let sp = 0, csp = 0, olen = 0, ip = 0, pc = 0, steps = 0;
  const done = (code: number): { out: Uint8Array; code: number } => ({ out: out.slice(0, olen), code });

  for (;;) {
    // One step, in the ISA's order: step limit, pc in range, known opcode, immediate fits.
    if (steps === 1_000_000) return done(10);
    if (pc < 0 || pc >= n) return done(9);
    const op = p[pc];
    if (!KNOWN(op)) return done(5);
    const imm = IMM(op);
    if (pc + imm >= n) return done(9);
    steps++;
    const next = pc + 1 + imm;
    const pp = PP[op] ?? (BIN(op) ? [2, 1] : [0, 0]);
    if (sp < pp[0]) return done(2);
    if (sp - pp[0] + pp[1] > 256) return done(3);
    pc = next;

    if (BIN(op)) {
      const b = D[sp - 1];
      const a = D[sp - 2];
      let r: number;
      if (op === 0x10) r = a + b;
      else if (op === 0x11) r = a - b;
      else if (op === 0x12) r = Math.imul(a, b); // the low 32 bits of the exact product
      else if (op === 0x13) { if (b === 0) return done(4); r = a === -2147483648 && b === -1 ? a : (a / b) | 0; }
      else if (op === 0x14) { if (b === 0) return done(4); r = b === -1 ? 0 : a % b; }
      else if (op === 0x15) { if (b === 0) return done(4); r = Math.floor((a >>> 0) / (b >>> 0)); }
      else if (op === 0x16) { if (b === 0) return done(4); r = (a >>> 0) % (b >>> 0); }
      else if (op === 0x18) r = a & b;
      else if (op === 0x19) r = a | b;
      else if (op === 0x1a) r = a ^ b;
      else if (op === 0x1c) r = a << (b & 31); // only the low 5 bits of the count
      else if (op === 0x1d) r = a >> (b & 31); // arithmetic shift
      else if (op === 0x1e) r = (a >>> (b & 31)) | 0; // logical shift, read back as signed
      else if (op === 0x20) r = a === b ? 1 : 0;
      else if (op === 0x21) r = a < b ? 1 : 0;
      else r = (a >>> 0) < (b >>> 0) ? 1 : 0; // LTU
      D[sp - 2] = r | 0;
      sp--;
      continue;
    }

    switch (op) {
      case 0x00: return done(0); // HALT
      case 0x01: D[sp++] = (p[next - 1] << 24) >> 24; break; // PUSH8, sign-extended
      case 0x02: D[sp++] = p[next - 4] | (p[next - 3] << 8) | (p[next - 2] << 16) | (p[next - 1] << 24); break; // PUSH32
      case 0x03: sp--; break; // POP
      case 0x04: D[sp] = D[sp - 1]; sp++; break; // DUP
      case 0x05: { const t = D[sp - 1]; D[sp - 1] = D[sp - 2]; D[sp - 2] = t; break; } // SWAP
      case 0x06: D[sp] = D[sp - 2]; sp++; break; // OVER
      case 0x07: { const a = D[sp - 3]; D[sp - 3] = D[sp - 2]; D[sp - 2] = D[sp - 1]; D[sp - 1] = a; break; } // ROT
      case 0x17: D[sp - 1] = -D[sp - 1]; break; // NEG, wrapping
      case 0x1b: D[sp - 1] = ~D[sp - 1]; break; // NOT
      case 0x30: case 0x31: case 0x32: { // JMP, JZ, JNZ: target checked at the next step
        const rel = ((p[next - 2] | (p[next - 1] << 8)) << 16) >> 16;
        let take = true;
        if (op !== 0x30) { const v = D[--sp]; take = op === 0x31 ? v === 0 : v !== 0; }
        if (take) pc = next + rel;
        break;
      }
      case 0x33: // CALL to an unsigned absolute address
        if (csp === 64) return done(7);
        C[csp++] = next;
        pc = p[next - 2] | (p[next - 1] << 8);
        break;
      case 0x34: // RET
        if (csp === 0) return done(8);
        pc = C[--csp];
        break;
      case 0x40: { // LOAD
        const addr = D[sp - 1];
        if (addr < 0 || addr > 255) return done(6);
        D[sp - 1] = M[addr];
        break;
      }
      case 0x41: { // STORE: ( v addr -- )
        const addr = D[sp - 1];
        if (addr < 0 || addr > 255) return done(6);
        M[addr] = D[sp - 2];
        sp -= 2;
        break;
      }
      case 0x50: { // PRINT the value in decimal ASCII; past the limit it writes nothing
        const s = String(D[sp - 1]);
        if (olen + s.length > 65536) return done(11);
        for (let i = 0; i < s.length; i++) out[olen++] = s.charCodeAt(i);
        sp--;
        break;
      }
      case 0x51: // PUTC one byte
        if (olen + 1 > 65536) return done(11);
        out[olen++] = D[--sp] & 255;
        break;
      case 0x52: D[sp++] = ip < inp.length ? inp[ip++] : -1; break; // IN: -1 once used up
      case 0x53: return done(D[sp - 1] & 255); // EXIT with the low byte
    }
  }
}

const inputDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const outputDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
for (const id of fs.readdirSync(inputDir).sort()) {
  const program = new Uint8Array(fs.readFileSync(path.join(inputDir, id, 'program.bin')));
  const input = new Uint8Array(fs.readFileSync(path.join(inputDir, id, 'input.bin')));
  const r = run(program, input);
  fs.mkdirSync(path.join(outputDir, id), { recursive: true });
  fs.writeFileSync(path.join(outputDir, id, 'stdout.bin'), r.out);
  fs.writeFileSync(path.join(outputDir, id, 'exit.txt'), `${r.code}\n`);
}
