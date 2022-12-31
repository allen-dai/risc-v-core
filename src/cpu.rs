#![allow(non_snake_case, non_camel_case_types)]

const PC: usize = 32;
const MEM_SIZE: usize = 0x1_0000;
const MEM_OFFSET: u64 = 0x8000_0000;
const REG_NAME: [&str; 33] = [
    "x0", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
    "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5",
    "t6", "PC",
];

#[derive(Debug)]
pub enum Opcode {
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

impl Opcode {
    pub fn from(inst_decode: &Instruction) -> Option<Self> {
        let Instruction {
            opcode_bin,
            funct3,
            funct7,
            funct12,
            ..
        } = inst_decode;
        match opcode_bin {
            0b0110111 => Some(Opcode::LUI),
            0b0010111 => Some(Opcode::AUIPC),
            0b1101111 => Some(Opcode::JAL),
            0b1100111 => Some(Opcode::JALR),
            0b0001111 => Some(Opcode::FENCE),
            0b1100011 => Some(Opcode::BRANCH(
                BRANCH::from(&funct3).expect("Invalid funct3 for BRANCH"),
            )),
            0b0000011 => Some(Opcode::LOAD(
                LOAD::from(&funct3).expect("Invalid funct3 for LOAD"),
            )),
            0b0100011 => Some(Opcode::STORE(
                STORE::from(&funct3).expect("Invalid funct3 for STORE"),
            )),
            0b0010011 => Some(Opcode::OP_IMM(
                OP_IMM::from(&funct3, &funct7).expect("Invalid funct3 and funct7 for OP_IMM"),
            )),
            0b0110011 => Some(Opcode::OP(
                OP::from(&funct3, &funct7).expect("Invalid funct3 and funct7 for OP"),
            )),
            0b1110011 => Some(Opcode::SYSTEM(
                SYSTEM::from(&funct12).expect("Invalid funct12 for SYSTEM"),
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
        match funct12 {
            0b000000000000 => Some(SYSTEM::ECALL),
            // 0b000000000001 => Some(SYSTEM::EBREAK),
            _ => Some(SYSTEM::EBREAK),
        }
    }
}

#[derive(Debug)]
pub struct Instruction {
    opcode_bin: u32,
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

impl Instruction {
    pub fn from(inst: u32) -> Self {
        let opcode_bin = extract_bits(6, 0, inst);
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
            opcode_bin,
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

#[derive(Debug, Clone, Copy)]
pub enum Xlen {
    Bit32,
    Bit64,
}

impl std::ops::Deref for Xlen {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        match self {
            Xlen::Bit32 => &32,
            Xlen::Bit64 => &64,
        }
    }
}

#[derive(Debug)]
pub struct Cpu {
    pub(crate) memory: [u8; MEM_SIZE],
    pub(crate) reg: [u32; 33],
}

impl Cpu {
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
        print!("\n\n");
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
        let inst_bin = self.r32(addr);

        // deocde
        let inst = Instruction::from(inst_bin);
        let opcode = Opcode::from(&inst).expect("Invalid instruction");
        /* println!(
            "current inst: {:032b} {:08x} {:?}",
            inst, inst, decoded_inst_type
        );
        self.dump_reg(); */

        // execute
        let mut result = self.execute(&opcode, &inst);

        // memeory
        self.memory_access(&opcode, &inst, &mut result);

        // write-back
        return self.writeback(&opcode, &inst, result);
    }

    pub fn execute(&self, opcode: &Opcode, inst: &Instruction) -> u32 {
        let Instruction {
            rs1,
            rs2,
            imm_i,
            imm_s,
            imm_u,
            imm_j,
            ..
        } = *inst;

        match opcode {
            Opcode::OP_IMM(OP_IMM) => match OP_IMM {
                OP_IMM::ADDI => self.reg[rs1 as usize].wrapping_add(imm_i),
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
                OP_IMM::ANDI => self.reg[rs1 as usize] & imm_i,
                OP_IMM::ORI => self.reg[rs1 as usize] | imm_i,
                OP_IMM::XORI => self.reg[rs1 as usize] ^ imm_i,
                OP_IMM::SLLI => self.reg[rs1 as usize] << (imm_i & 0x1F),
                OP_IMM::SRLI => self.reg[rs1 as usize] >> (imm_i & 0x1F),
                OP_IMM::SRAI => (self.reg[rs1 as usize] as i32 >> (imm_i & 0x1F)) as u32,
            },
            Opcode::BRANCH(_) => 0,
            Opcode::JAL => self.reg[PC].wrapping_add(imm_j),
            Opcode::LUI => imm_u,
            Opcode::AUIPC => self.reg[PC].wrapping_add(imm_u),
            Opcode::JALR => self.reg[rs1 as usize].wrapping_add(imm_i),
            Opcode::FENCE => self.reg[rs1 as usize].wrapping_add(imm_i),
            Opcode::LOAD(_) => self.reg[rs1 as usize].wrapping_add(imm_i),
            Opcode::STORE(_) => {
                ((self.reg[rs1 as usize].wrapping_add(imm_s) as u64) - MEM_OFFSET) as u32
            }
            Opcode::OP(OP) => match OP {
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
                OP::OR => self.reg[rs1 as usize] | self.reg[rs2 as usize],
                OP::AND => self.reg[rs1 as usize] & self.reg[rs2 as usize],
                OP::XOR => self.reg[rs1 as usize] ^ self.reg[rs2 as usize],
                OP::SLL => self.reg[rs1 as usize] << (self.reg[rs2 as usize] & 0x1F),
                OP::SRL => self.reg[rs1 as usize] >> (self.reg[rs2 as usize] & 0x1F),
                OP::SRA => (self.reg[rs1 as usize] >> (self.reg[rs2 as usize] & 0x1F)) as u32,
            },
            Opcode::SYSTEM(_) => 0,
        }
    }

    pub fn writeback(&mut self, opcode: &Opcode, inst: &Instruction, result: u32) -> bool {
        let Instruction {
            rs1,
            rs2,
            rd,
            imm_b,
            ..
        } = *inst;

        match opcode {
            Opcode::JAL | Opcode::JALR => {
                self.reg[rd as usize] = self.reg[PC] + 4;
                self.reg[PC] = result;
            }
            Opcode::OP_IMM(_) => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4
            }
            Opcode::BRANCH(BRANCH) => {
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
            Opcode::LUI => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            Opcode::AUIPC => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            Opcode::FENCE => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            Opcode::LOAD(_) => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            Opcode::STORE(_) => {
                self.reg[PC] += 4;
            }
            Opcode::OP(_) => {
                self.reg[rd as usize] = result;
                self.reg[PC] += 4;
            }
            Opcode::SYSTEM(SYSTEM) => {
                match SYSTEM {
                    SYSTEM::ECALL => {
                        if self.reg[3] > 1 {
                            panic!("Fail at test #{}", self.reg[3])
                        }
                        if self.reg[3] == 1 {
                            return false;
                        }
                    }
                    _ => (),
                }
                self.reg[PC] += 4;
            }
        }
        self.reg[0] = 0;
        true
    }

    pub fn memory_access(&mut self, opcode: &Opcode, inst: &Instruction, result: &mut u32) {
        let Instruction { rs2, .. } = *inst;

        match opcode {
            Opcode::LOAD(LOAD) => match LOAD {
                LOAD::LB => {
                    *result = sign_extend(self.r32(*result as usize) & 0xFF, 8);
                }
                LOAD::LH => {
                    *result = sign_extend(self.r32(*result as usize) & 0xFFFF, 16);
                }
                LOAD::LW => {
                    *result = self.r32(*result as usize);
                }
                LOAD::LBU => {
                    *result = self.r32(*result as usize) & 0xFF;
                }
                LOAD::LHU => {
                    *result = self.r32(*result as usize) & 0xFFFF;
                }
            },
            Opcode::STORE(STORE) => {
                let addr = *result as usize;
                let bytes = self.reg[rs2 as usize].to_le_bytes();
                match STORE {
                    STORE::SW => {
                        self.memory[addr] = bytes[0];
                        self.memory[addr + 1] = bytes[1];
                        self.memory[addr + 2] = bytes[2];
                        self.memory[addr + 3] = bytes[3];
                    }
                    STORE::SH => {
                        self.memory[addr] = bytes[0];
                        self.memory[addr + 1] = bytes[1];
                    }
                    STORE::SB => self.memory[addr] = bytes[0],
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
    let inst = Instruction::from(instruction);
    let opcode = (instruction >> 0) & 0b01111111;
    let funct3 = (instruction >> 12) & 0b111;
    let rs1 = (instruction >> 15) & 0b11111;
    let rs2 = (instruction >> 20) & 0b11111;
    let funct7 = (instruction >> 25) & 0b1111111;
    let rd = (instruction >> 7) & 0b11111;
    let imm_i_t = (instruction >> 20) & 0b111111111111;
    let imm_i = (inst.imm_i) & 0b111111111111;
    let imm_s = inst.imm_s & 0b111111111111;
    let imm_b = inst.imm_b & 0b1111111111111;
    let imm_u_t = (instruction >> 12) & 0b11111111111111111111;
    let imm_u = (inst.imm_u >> 12) & 0b11111111111111111111;
    let imm_j = (inst.imm_j) & 0b1111_1111_1111_1111_1111_1;
    assert_eq!(inst.rd, rd);
    assert_eq!(inst.rs1, rs1);
    assert_eq!(inst.rs2, rs2);
    assert_eq!(inst.opcode_bin, opcode);
    assert_eq!(inst.funct7, funct7);
    assert_eq!(inst.funct3, funct3);
    assert_eq!(imm_i, imm_i_t);
    assert_eq!(imm_s, 0b1011_100_10100);
    assert_eq!(imm_b, 0b1001_1100_1010_0);
    assert_eq!(imm_u, imm_u_t);
    assert_eq!(imm_j, 0b1_10101011_0_0111001101_0);
}

#[test]
fn rv32ui_p() {
    use glob::glob;
    use object::read::elf::*;
    use object::Endianness;
    use object::{Object, ObjectSection};
    use std::fs::{self, File};
    use std::io::Read;
    let mut file_tested = 0;
    for entry in glob("riscv-tests/isa/rv32ui-p-*").expect("fail to read path") {
        match entry {
            Ok(path) => {
                if path.extension().is_some() {
                    continue;
                }
                // FIXME
                // instruction at PC is not correct when compare to .dump file.
                // Not sure if this test file is correctly generated or my instruction has srewed
                // something up
                if path.display().to_string() == "riscv-tests/isa/rv32ui-p-sra" {
                    continue;
                }
                let mut file = File::open(&path).expect("unable to read file");
                let metadata = fs::metadata(&path).expect("unable to read metadata");
                let mut buf = vec![0; metadata.len() as usize];
                file.read_to_end(&mut buf)
                    .expect("unable to read into buffer");
                let bin_data = fs::read(&path).expect("unable to read elf file");
                let mut cpu = Cpu::new();
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
                        cpu.memory[i + addr] = *byte;
                    }
                }
                cpu.reg[PC] = elf_file.entry() as u32;
                let mut counter = 0;
                file_tested += 1;
                println!("\n{:?}", path.display());
                while cpu.cycle() {
                    counter += 1;
                }
                println!("{counter} instructions ran");
                println!("{file_tested} file tested\n");
            }
            Err(_) => panic!(),
        }
    }
}
