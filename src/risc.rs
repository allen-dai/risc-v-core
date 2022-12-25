#![allow(non_snake_case, unused, non_camel_case_types)]

use std::collections::HashMap;

use anyhow::Result;

const PC: usize = 32;
const MEM_SIZE: usize = 0x1_0000;
const MEM_OFFSET: u64 = 0x8000_0000;

#[derive(Debug)]
pub enum RV32I_Opcode {
    LUI,
    AUIPC,
    JAL,
    JALR,
    FENCE,
    BRANCH(BRANCH), // BEQ BNE BLT BGE BLTU BGEU
    LOAD(LOAD),     // LB LH LW LBU LHU
    STORE(STORE),   // SB SH SW
    OP_IMM(OP_IMM), // ADDI SLTI SLTIU XORI ORI ANDI SLLI SRLI SRAI
    OP(OP),         // ADD SUB SLL SLT SLTU XOR SRL SRA OR AND
    SYSTEM(SYSTEM), // ECALL EBREAK
}

impl RV32I_Opcode {
    pub fn from(inst_decode: &Inst_Info) -> Option<Self> {
        let Inst_Info {
            opcode,
            rs1,
            rs2,
            rd,
            imm_i,
            imm_s,
            imm_b,
            imm_u,
            imm_j,
            funct3,
            funct7,
            funct12,
        } = inst_decode;
        match opcode {
            0b0110111 => Some(RV32I_Opcode::LUI),
            0b0010111 => Some(RV32I_Opcode::AUIPC),
            0b1101111 => Some(RV32I_Opcode::JAL),
            0b1100111 => Some(RV32I_Opcode::JALR),
            0b0001111 => Some(RV32I_Opcode::FENCE),
            0b1100011 => Some(RV32I_Opcode::BRANCH(
                BRANCH::from(&funct3).expect("Invalid funct3 for BRANCH"),
            )),
            0b0000011 => Some(RV32I_Opcode::LOAD(
                LOAD::from(&funct3).expect("Invalid funct3 for LOAD"),
            )),
            0b0100011 => Some(RV32I_Opcode::STORE(
                STORE::from(&funct3).expect("Invalid funct3 for STORE"),
            )),
            0b0010011 => Some(RV32I_Opcode::OP_IMM(
                OP_IMM::from(&funct3, &funct7).expect("Invalid funct3 and funct7 for OP_IMM"),
            )),
            0b0110011 => Some(RV32I_Opcode::OP(
                OP::from(&funct3, &funct7).expect("Invalid funct3 and funct7 for OP_IMM"),
            )),
            0b1110011 => Some(RV32I_Opcode::SYSTEM(
                SYSTEM::from(&funct12).expect("Invalid imm_i for SYSTEM"),
            )),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum BRANCH {
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
}

impl BRANCH {
    pub fn from(funct3: &u32) -> Option<Self> {
        match funct3 {
            0b000 => Some(BRANCH::BEQ),
            0b001 => Some(BRANCH::BNE),
            0b100 => Some(BRANCH::BLT),
            0b101 => Some(BRANCH::BGE),
            0b110 => Some(BRANCH::BLTU),
            0b111 => Some(BRANCH::BGEU),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum LOAD {
    LB,
    LH,
    LW,
    LBU,
    LHU,
}

impl LOAD {
    pub fn from(funct3: &u32) -> Option<Self> {
        match funct3 {
            0b000 => Some(LOAD::LB),
            0b001 => Some(LOAD::LH),
            0b010 => Some(LOAD::LW),
            0b100 => Some(LOAD::LBU),
            0b101 => Some(LOAD::LHU),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum STORE {
    SB,
    SH,
    SW,
}

impl STORE {
    pub fn from(funct3: &u32) -> Option<Self> {
        match funct3 {
            0b000 => Some(STORE::SB),
            0b001 => Some(STORE::SH),
            0b010 => Some(STORE::SW),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum OP {
    ADD,
    SUB,
    SLL,
    SLT,
    SLTU,
    XOR,
    SRL,
    SRA,
    OR,
    AND,
}

impl OP {
    pub fn from(funct3: &u32, funct7: &u32) -> Option<Self> {
        match funct3 {
            0b000 => match funct7 {
                0b0000000 => Some(OP::ADD),
                0b0100000 => Some(OP::SUB),
                _ => None,
            },
            0b001 => Some(OP::SLL),
            0b010 => Some(OP::SLT),
            0b011 => Some(OP::SLTU),
            0b100 => Some(OP::XOR),
            0b101 => match funct7 {
                0b0000000 => Some(OP::SRL),
                0b0100000 => Some(OP::SRA),
                _ => None,
            },
            0b110 => Some(OP::OR),
            0b111 => Some(OP::AND),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum OP_IMM {
    ADDI,
    SLTI,
    SLTIU,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
}

impl OP_IMM {
    pub fn from(funct3: &u32, funct7: &u32) -> Option<Self> {
        match funct3 {
            0b000 => Some(OP_IMM::ADDI),
            0b010 => Some(OP_IMM::SLTI),
            0b011 => Some(OP_IMM::SLTIU),
            0b100 => Some(OP_IMM::XORI),
            0b110 => Some(OP_IMM::ORI),
            0b111 => Some(OP_IMM::ANDI),
            0b001 => Some(OP_IMM::SLLI),
            0b101 => match funct7 {
                0b0000000 => Some(OP_IMM::SRLI),
                0b0100000 => Some(OP_IMM::SRAI),
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum SYSTEM {
    ECALL,
    EBREAK,
}

impl SYSTEM {
    pub fn from(funct12: &u32) -> Option<Self> {
        match funct12 & 0b1 {
            0b0 => Some(SYSTEM::ECALL),
            0b1 => Some(SYSTEM::EBREAK),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct Inst_Info {
    opcode: u32,
    rs1: u32,
    rs2: u32,
    rd: u32,
    imm_i: u32,
    imm_s: u32,
    imm_b: u32,
    imm_u: u32,
    imm_j: u32,
    funct3: u32,
    funct7: u32,
    funct12: u32,
}

impl Inst_Info {
    pub fn from(inst: u32) -> Self {
        let opcode = extract_bits(6, 0, inst);
        let funct7 = extract_bits(31, 25, inst);
        let rs2 = extract_bits(24, 20, inst);
        let rs1 = extract_bits(19, 15, inst);
        let funct3 = extract_bits(14, 12, inst);
        let rd = extract_bits(11, 7, inst);

        let imm_i = sign_extend(extract_bits(31, 20, inst), 12);
        let imm_s = sign_extend(
            extract_bits(31, 25, inst) << 5 | extract_bits(11, 7, inst),
            12,
        );
        let imm_b = sign_extend(
            extract_bits(31, 31, inst) << 12
                | extract_bits(30, 25, inst) << 5
                | extract_bits(11, 8, inst) << 1
                | extract_bits(7, 7, inst) << 11,
            13,
        );
        let imm_u = extract_bits(31, 12, inst) << 12;
        let imm_j = sign_extend(
            extract_bits(31, 31, inst) << 20
                | extract_bits(30, 21, inst) << 1
                | extract_bits(20, 20, inst) << 11
                | extract_bits(19, 12, inst) << 12,
            21,
        );
        let funct12 = sign_extend(extract_bits(31, 20, inst), 12);

        Self {
            opcode,
            rs1,
            rs2,
            rd,
            imm_i,
            imm_s,
            imm_b,
            imm_u,
            imm_j,
            funct3,
            funct7,
            funct12,
        }
    }
}

const REG_NAME: [&str; 33] = [
    "x0", "ra", "sp", "gp", "tp", "t0", "t1", "x1", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5",
    "t6", "PC",
];

#[derive(Debug)]
pub struct RV32I {
    pub(crate) memory: [u8; MEM_SIZE],
    pub(crate) reg: [u32; 33],
}

impl RV32I {
    pub fn new() -> Self {
        Self {
            memory: [0; MEM_SIZE],
            reg: [0; 33],
        }
    }

    pub fn dump_reg(&self) {
        let mut c = 0;
        let mut i = 0;
        for reg in self.reg {
            if c == 1 {
                print!("\n");
                c = 0;
            }
            print!("{:>3}: {:08x} ", REG_NAME[i], reg);
            i += 1;
            c += 1;
        }
        print!("\n");
    }

    pub fn r32(&self, mut addr: usize) -> u32 {
        addr -= MEM_OFFSET as usize;
        u32::from_le_bytes([
            self.memory[addr],
            self.memory[addr + 1],
            self.memory[addr + 2],
            self.memory[addr + 3],
        ])
    }

    pub fn cycle(&mut self) -> bool {
        // fetch
        let addr = self.reg[PC] as usize;
        let inst = self.r32(addr);

        // deocde
        let inst_info = Inst_Info::from(inst);
        let decoded_inst_type = RV32I_Opcode::from(&inst_info).expect("Invalid instruction");
        println!(
            "current inst: {:032b} {:08x} {:?}",
            inst, inst, decoded_inst_type
        );

        self.dump_reg();

        // execute
        let result = self.execute(&decoded_inst_type, &inst_info);

        // memeory
        self.memory_access(&decoded_inst_type, &inst_info, result);
        // write-back
        return self.writeback(&decoded_inst_type, &inst_info, result);
    }

    pub fn execute(&self, inst_type: &RV32I_Opcode, inst_info: &Inst_Info) -> u32 {
        let Inst_Info {
            opcode,
            rs1,
            rs2,
            rd,
            imm_i,
            imm_s,
            imm_b,
            imm_u,
            imm_j,
            funct3,
            funct7,
            funct12,
        } = *inst_info;

        match inst_type {
            RV32I_Opcode::OP_IMM(OP_IMM) => {
                match OP_IMM {
                    // ADDI adds the sign-extended 12-bit immediate to register rs1.
                    // Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.
                    // ADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.
                    OP_IMM::ADDI => self.reg[rs1 as usize].wrapping_add(imm_i),
                    // SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less
                    // than the sign- extended immediate when both are treated as signed numbers, else 0 is written to rd.
                    // SLTIU is similar but compares the values as unsigned numbers
                    // (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number).
                    // Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).
                    OP_IMM::SLTI => {
                        if (self.reg[rs1 as usize] as i32) < (imm_i as i32) {
                            1
                        } else {
                            0
                        }
                    }
                    OP_IMM::SLTIU => {
                        if self.reg[rs1 as usize] < imm_i {
                            1
                        } else {
                            0
                        }
                    }
                    // ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR
                    // on register rs1 and the sign-extended 12-bit immediate and place the result in rd.
                    // Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).
                    OP_IMM::ANDI => self.reg[rs1 as usize] & imm_i,
                    OP_IMM::ORI => self.reg[rs1 as usize] | imm_i,
                    OP_IMM::XORI => self.reg[rs1 as usize] ^ imm_i,
                    // Shifts by a constant are encoded as a specialization of the I-type format.
                    // The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field.
                    // The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits);
                    // SRLI is a logical right shift (zeros are shifted into the upper bits);
                    // and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).
                    OP_IMM::SLLI => self.reg[rs1 as usize] << (imm_i & 0x1F),
                    OP_IMM::SRLI => self.reg[rs1 as usize] >> (imm_i & 0x1F),
                    OP_IMM::SRAI => (self.reg[rs1 as usize] as i32 >> (imm_i & 0x1F)) as u32,
                }
            }
            RV32I_Opcode::BRANCH(_) => 0,
            RV32I_Opcode::JAL => self.reg[PC] + imm_j,
            RV32I_Opcode::LUI => imm_u,
            // AUIPC (add upper immediate to pc) is used to build pc-relative addresses
            // and uses the U-type format. AUIPC forms a 32-bit offset from the 20-bit U-immediate,
            // filling in the lowest 12 bits with zeros, adds this offset to the address of the AUIPC instruction,
            // then places the result in register rd.
            RV32I_Opcode::AUIPC => self.reg[PC] + imm_u,
            // The indirect jump instruction JALR (jump and link register) uses the I-type encoding.
            // The target address is obtained by adding the sign-extended 12-bit I-immediate to the register rs1,
            // then setting the least-significant bit of the result to zero.
            // The address of the instruction following the jump (pc+4) is written to register rd.
            // Register x0 can be used as the destination if the result is not required.
            RV32I_Opcode::JALR => self.reg[rs1 as usize].wrapping_add(imm_i),

            // TODO Need to figure out how to implement this correctly...
            RV32I_Opcode::FENCE => self.reg[rs1 as usize].wrapping_add(imm_i),
            RV32I_Opcode::LOAD(LOAD) => match LOAD {
                LOAD::LB => todo!(),
                LOAD::LH => todo!(),
                LOAD::LW => todo!(),
                LOAD::LBU => todo!(),
                LOAD::LHU => todo!(),
            },
            // The SW, SH, and SB instructions store 32-bit, 16-bit, and 8-bit values from the low bits of register rs2 to memory.
            RV32I_Opcode::STORE(STORE) => match STORE {
                STORE::SW => self.reg[rs2 as usize],
                STORE::SH => self.reg[rs2 as usize] & 0xFF,
                STORE::SB => self.reg[rs2 as usize] & 0xF,
            },
            RV32I_Opcode::OP(OP) => match OP {
                OP::ADD => self.reg[rs1 as usize].wrapping_add(self.reg[rs2 as usize]),
                OP::SUB => self.reg[rs1 as usize].wrapping_sub(self.reg[rs2 as usize]),
                OP::SLT => {
                    if (self.reg[rs1 as usize] as i32) < (self.reg[rs2 as usize] as i32) {
                        1
                    } else {
                        0
                    }
                }
                OP::SLTU => {
                    if self.reg[rs1 as usize] < self.reg[rs2 as usize] {
                        1
                    } else {
                        0
                    }
                }
                OP::SLL => todo!(),
                OP::XOR => todo!(),
                OP::SRL => todo!(),
                OP::SRA => todo!(),
                OP::OR => todo!(),
                OP::AND => todo!(),
            },
            RV32I_Opcode::SYSTEM(SYSTEM) => match SYSTEM {
                SYSTEM::ECALL => 0,
                SYSTEM::EBREAK => 0,
            },
        }
    }

    pub fn writeback(&mut self, inst_type: &RV32I_Opcode, inst_info: &Inst_Info, result: u32) -> bool {
        let Inst_Info {
            opcode,
            rs1,
            rs2,
            rd,
            imm_i,
            imm_s,
            imm_b,
            imm_u,
            imm_j,
            funct3,
            funct7,
            funct12,
        } = *inst_info;

        match inst_type {
            RV32I_Opcode::JAL | RV32I_Opcode::JALR => {
                self.reg[rd as usize] = self.reg[PC] + 4;
                self.reg[PC] = result;
            }
            RV32I_Opcode::OP_IMM(_) => {
                //println!("{}", result);
                self.reg[rd as usize] = result;
                self.reg[PC] += 4
            }
            RV32I_Opcode::BRANCH(BRANCH) => {
                // println!("branch {:032b}", imm_b);
                let mut new_pc = 4;
                match BRANCH {
                    BRANCH::BEQ => {
                        if (self.reg[rs1 as usize] as i32) == (self.reg[rs2 as usize] as i32) {
                            new_pc = imm_b;
                        }
                    }
                    BRANCH::BNE => {
                        if (self.reg[rs1 as usize] as i32) != (self.reg[rs2 as usize] as i32) {
                            new_pc = imm_b;
                        }
                    }
                    BRANCH::BLT => {
                        if (self.reg[rs1 as usize] as i32) < (self.reg[rs2 as usize] as i32) {
                            new_pc = imm_b;
                        }
                    }
                    BRANCH::BGE => {
                        if (self.reg[rs1 as usize] as i32) >= (self.reg[rs2 as usize] as i32) {
                            new_pc = imm_b;
                        }
                    }
                    BRANCH::BLTU => {
                        if self.reg[rs1 as usize] < self.reg[rs2 as usize] {
                            new_pc = imm_b;
                        }
                    }
                    BRANCH::BGEU => {
                        if self.reg[rs1 as usize] >= self.reg[rs2 as usize] {
                            new_pc = imm_b;
                        }
                    }
                }
                self.reg[PC] = self.reg[PC].wrapping_add(new_pc);
            }
            RV32I_Opcode::LUI => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            RV32I_Opcode::AUIPC => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            RV32I_Opcode::FENCE => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            RV32I_Opcode::LOAD(_) => todo!(),
            RV32I_Opcode::STORE(_) => {}
            RV32I_Opcode::OP(OP) => {
                match OP {
                    OP::ADD => {
                        self.reg[rd as usize] = result;
                    }
                    OP::SUB => todo!(),
                    OP::SLL => todo!(),
                    OP::SLT => todo!(),
                    OP::SLTU => todo!(),
                    OP::XOR => todo!(),
                    OP::SRL => todo!(),
                    OP::SRA => todo!(),
                    OP::OR => todo!(),
                    OP::AND => todo!(),
                }
                self.reg[PC] += 4;
            }
            RV32I_Opcode::SYSTEM(SYSTEM) => {
                match SYSTEM {
                    SYSTEM::ECALL => {
                        if self.reg[3] > 1{
                            panic!("Fail test")
                        }
                        else if self.reg[3] == 1{
                            return false
                        }
                    }
                    _ => ()
                }
                self.reg[PC] += 4;
            }
        }
        self.reg[0] = 0;

        true
    }

    pub fn memory_access(
        &mut self,
        decoded_inst_type: &RV32I_Opcode,
        inst_info: &Inst_Info,
        result: u32,
    ) {
        let Inst_Info {
            opcode,
            rs1,
            rs2,
            rd,
            imm_i,
            imm_s,
            imm_b,
            imm_u,
            imm_j,
            funct3,
            funct7,
            funct12,
        } = *inst_info;

        match decoded_inst_type {
            RV32I_Opcode::LOAD(_) => todo!(),
            RV32I_Opcode::STORE(STORE) => {
                println!("\n\n{} {}", self.reg[rs1 as usize], imm_s);
                println!("{:032b} {:032b}\n\n", self.reg[rs1 as usize], imm_s);
                let addr = (self.reg[rs1 as usize].wrapping_add(imm_s) as u64) as usize;
                println!("{:032b}", addr as i32);
                match STORE {
                    STORE::SW => {
                        self.memory[addr] = (self.reg[rs2 as usize] & 0xF000) as u8;
                        self.memory[addr + 1] = (self.reg[rs2 as usize] & 0x0F00) as u8;
                        self.memory[addr + 2] = (self.reg[rs2 as usize] & 0x00F0) as u8;
                        self.memory[addr + 3] = (self.reg[rs2 as usize] & 0x000F) as u8;
                    }
                    STORE::SH => {
                        self.memory[addr] = (self.reg[rs2 as usize] & 0x00F0) as u8;
                        self.memory[addr + 1] = (self.reg[rs2 as usize] & 0x000F) as u8;
                    }
                    STORE::SB => self.memory[addr] = (self.reg[rs2 as usize] & 0xF) as u8,
                }
            }
            _ => (),
        }
    }
}

pub fn extract_bits(s: u32, e: u32, inst: u32) -> u32 {
    inst >> e & (1 << (s - e + 1)) - 1
}

pub fn sign_extend(n: u32, bit: usize) -> u32 {
    let offset = 32 - bit;
    if n >= u32::MAX {
        0
    } else {
        ((n as i32) << offset >> offset) as u32
    }
}

#[test]
fn sign_ext_test() {
    let val = 0b1000_0000_1111;
    let ext = sign_extend(val, 12);
    assert_eq!(ext, 0b1111_1111_1111_1111_1111_1000_0000_1111);
}

#[test]
fn extract_bits_test() {
    let val = 0b1100_0000_1111;
    let t1 = extract_bits(11, 10, val);
    let t2 = extract_bits(11, 11, val);
    assert_eq!(t1, 0b11);
    assert_eq!(t2, 0b1);
}

#[test]
fn inst_decode_test() {
    let instruction = 0b1011_1001_1010_1010_1011_1010_0011_0011;
    let inst = Inst_Info::from(instruction);
    let opcode = (instruction >> 0) & 0b01111111;
    let funct3 = (instruction >> 12) & 0b111;
    let rs1 = (instruction >> 15) & 0b11111;
    let rs2 = (instruction >> 20) & 0b11111;
    let funct7 = (instruction >> 25) & 0b1111111;
    let rd = (instruction >> 7) & 0b11111;
    let imm_i_t = (instruction >> 20) & 0b111111111111;
    let imm_i = (inst.imm_i) & 0b111111111111;
    let imm_s_t = ((instruction >> 25) & 0b111) | ((instruction >> 7) & 0b1111111);
    let imm_s = inst.imm_s & 0b111111111111;
    let imm_b = inst.imm_b & 0b1111111111111;
    let imm_u_t = (instruction >> 12) & 0b11111111111111111111;
    let imm_u = (inst.imm_u >> 12) & 0b11111111111111111111;
    let imm_j = (inst.imm_j) & 0b1111_1111_1111_1111_1111_1;
    let imm_j_t = ((instruction >> 21) & 0b1111)
        | ((instruction >> 20) & 0b1)
        | ((instruction >> 12) & 0b11111111111)
        | ((instruction >> 11) & 0b1);
    assert_eq!(inst.rd, rd);
    assert_eq!(inst.rs1, rs1);
    assert_eq!(inst.rs2, rs2);
    assert_eq!(inst.opcode, opcode);
    assert_eq!(inst.funct7, funct7);
    assert_eq!(inst.funct3, funct3);
    assert_eq!(imm_i, imm_i_t);
    assert_eq!(imm_s, 0b1011_100_10100);
    assert_eq!(imm_b, 0b1001_1100_1010_0);
    //println!("{:032b}\n{:032b}\n{:032b}", inst.imm_u, imm_u, imm_u_t);
    assert_eq!(imm_u, imm_u_t);
    //println!("{:032b}\n{:032b}\n{:032b}", inst.imm_j, imm_j, 0b1_10101011_0_0111001101_0);
    assert_eq!(imm_j, 0b1_10101011_0_0111001101_0);
}

#[test]
fn rv32ui() {
    use glob::glob;
    use object::read::elf::*;
    use object::Endianness;
    use object::{Object, ObjectSection};
    use std::fs::{self, File};
    use std::io::Read;
    for entry in glob("riscv-tests/isa/rv32ui-p-add").expect("fail to read path") {
        match entry {
            Ok(path) => {
                if !path.extension().is_some() {
                    let mut file = File::open(&path).expect("unable to read file");
                    let metadata = fs::metadata(&path).expect("unable to read metadata");
                    let mut buf = vec![0; metadata.len() as usize];
                    file.read_to_end(&mut buf)
                        .expect("unable to read into buffer");
                    // println!("\n{:?}", path.display());
                    let bin_data = fs::read(&path).expect("unable to read elf file");
                    let mut rv32i = RV32I::new();
                    let elf_file = ElfFile32::<Endianness>::parse(&*bin_data)
                        .expect("unable to parse elf raw bin file");
                    /* for segment in elf_file.segments() {
                        println!("{:?}", segment);
                    } */
                    for section in elf_file.sections() {
                        if section.address() < MEM_OFFSET {
                            continue;
                        }
                        let addr = (section.address() - MEM_OFFSET) as usize;
                        for (i, byte) in section.data().unwrap().iter().enumerate() {
                            rv32i.memory[i + addr] = *byte;
                        }
                    }
                    rv32i.reg[PC] = elf_file.entry() as u32;
                    let mut counter = 0;
                    while rv32i.cycle() {
                        counter += 1;
                        println!("{counter} instruction tested\n");
                    }
                }
            }
            Err(_) => panic!(),
        }
    }
}
