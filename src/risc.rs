#![allow(non_snake_case)]
use std::io::Cursor;

use anyhow::Result;

const PC: usize = 32;
const MEM_SIZE: usize = 0x1_0000;

pub enum RV32I_Opcode {
    LUI,
    AUIPC,
    JAL,
    JALR,
    FENCE,
    BRANCH, // BEQ BNE BLT BGE BLTU BGEU
    LOAD,   // LB LH LW LBU LHU
    STORE,  // SB SH SW
    OP,     // ADDI SLTI SLTIU XORI ORI ANDI SLLI SRLI SRAI
    SYSTEM, // ECALL EBREAK
}

pub enum BRANCH {
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,
}

impl BRANCH {
    pub fn from(funct3: u32) -> Option<Self> {
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

pub enum OP {
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

impl OP {
    pub fn from(funct3: u32, funct7: u32) -> Option<Self> {
        match funct3 {
            0b000 => Some(OP::ADDI),
            0b010 => Some(OP::SLTI),
            0b011 => Some(OP::SLTIU),
            0b100 => Some(OP::XORI),
            0b110 => Some(OP::ORI),
            0b111 => Some(OP::ANDI),
            0b001 => Some(OP::SLLI),
            0b101 => match funct7 {
                0b0000000 => Some(OP::SRLI),
                0b0100000 => Some(OP::SRAI),
                _ => None,
            },
            _ => None,
        }
    }
}

impl RV32I_Opcode {
    pub fn from_inst(inst: u32) -> Option<Self> {
        // "opcode" value 6 to 0
        println!("extracted: {:06b}", extract_bits(6, 0, inst));
        match extract_bits(6, 0, inst) {
            0b0110111 => Some(RV32I_Opcode::LUI),
            0b0010111 => Some(RV32I_Opcode::AUIPC),
            0b1101111 => Some(RV32I_Opcode::JAL),
            0b1100111 => Some(RV32I_Opcode::JALR),
            0b0001111 => Some(RV32I_Opcode::FENCE),
            0b1100011 => Some(RV32I_Opcode::BRANCH),
            0b0000011 => Some(RV32I_Opcode::LOAD),
            0b0100011 => Some(RV32I_Opcode::STORE),
            0b0110011 => Some(RV32I_Opcode::OP),
            0b1110011 => Some(RV32I_Opcode::SYSTEM),
            _ => {
                //println!("Invalid instruction: {:032b}", inst);
                None
            }
        }
    }
}

pub enum OPS_TYPE {
    OP(OP),
    BRANCH(BRANCH),
}

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

    pub fn r32(&self, addr: usize) -> u32 {
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
        let opcode = RV32I_Opcode::from_inst(inst).expect("Invalid opcode");
        let mut funct7 = extract_bits(31, 25, inst);
        let mut rs2 = extract_bits(24, 20, inst);
        let mut rs1 = extract_bits(19, 15, inst);
        let mut funct3 = extract_bits(14, 12, inst);
        let mut rd = extract_bits(11, 7, inst);

        let mut imm_i = extract_bits(31, 20, inst);
        let mut imm_s = extract_bits(31, 25, inst) << 5 | extract_bits(11, 7, inst);
        let mut imm_b = extract_bits(31, 31, inst) << 12
            | extract_bits(30, 25, inst) << 5
            | extract_bits(11, 8, inst) << 1
            | extract_bits(7, 7, inst) << 11;
        let mut imm_u = extract_bits(31, 12, inst) << 12;
        let mut imm_j = extract_bits(31, 31, inst) << 20
            | extract_bits(30, 21, inst) << 1
            | extract_bits(20, 20, inst) << 11
            | extract_bits(19, 12, inst) << 12;

        let decode = (
            funct7, rs2, rs1, funct3, rd, imm_i, imm_s, imm_b, imm_u, imm_j,
        );

        let inst_type = match opcode {
            // I-type instruction
            RV32I_Opcode::OP => {
                OPS_TYPE::OP(OP::from(funct3, funct7).expect("Invalid Operation inst"))
            }
            RV32I_Opcode::BRANCH => {
                OPS_TYPE::BRANCH(BRANCH::from(funct3).expect("Invalid Branch inst"))
            }
            RV32I_Opcode::SYSTEM => return false,
            _ => todo!(""),
        };
        // println!("{:?}", &decode);
        // execute
        self.execute(inst_type, decode);
        // memeory
        // write-back

        true
    }

    pub fn execute(
        &mut self,
        inst_type: OPS_TYPE,
        decode: (u32, u32, u32, u32, u32, u32, u32, u32, u32, u32),
    ) {
        let (funct7, rs2, rs1, funct3, rd, imm_i, imm_s, imm_b, imm_u, imm_j) = decode;
        match inst_type {
            OPS_TYPE::OP(I_TYPE) => {
                match I_TYPE {
                    // ADDI adds the sign-extended 12-bit immediate to register rs1.
                    // Arithmetic overflow is ignored and the result is simply the low XLEN bits of the result.
                    // ADDI rd, rs1, 0 is used to implement the MV rd, rs1 assembler pseudoinstruction.
                    OP::ADDI => {
                        self.reg[rs1 as usize] = self.reg[rs1 as usize].wrapping_add(imm_i);
                    }
                    // SLTI (set less than immediate) places the value 1 in register rd if register rs1 is less
                    // than the sign- extended immediate when both are treated as signed numbers, else 0 is written to rd.
                    // SLTIU is similar but compares the values as unsigned numbers
                    // (i.e., the immediate is first sign-extended to XLEN bits then treated as an unsigned number).
                    // Note, SLTIU rd, rs1, 1 sets rd to 1 if rs1 equals zero, otherwise sets rd to 0 (assembler pseudoinstruction SEQZ rd, rs).
                    OP::SLTI => {
                        self.reg[rd as usize] = if (self.reg[rs1 as usize] as i32) < (imm_i as i32)
                        {
                            1
                        } else {
                            0
                        };
                    }
                    OP::SLTIU => {
                        self.reg[rd as usize] = if self.reg[rs1 as usize] < imm_i { 1 } else { 0 };
                    }
                    // ANDI, ORI, XORI are logical operations that perform bitwise AND, OR, and XOR
                    // on register rs1 and the sign-extended 12-bit immediate and place the result in rd.
                    // Note, XORI rd, rs1, -1 performs a bitwise logical inversion of register rs1 (assembler pseudoinstruction NOT rd, rs).
                    OP::ANDI => {
                        self.reg[rd as usize] = self.reg[rs1 as usize] & imm_i;
                    }
                    OP::ORI => {
                        self.reg[rd as usize] = self.reg[rs1 as usize] | imm_i;
                    }
                    OP::XORI => {
                        self.reg[rd as usize] = self.reg[rs1 as usize] ^ imm_i;
                    }
                    // Shifts by a constant are encoded as a specialization of the I-type format.
                    // The operand to be shifted is in rs1, and the shift amount is encoded in the lower 5 bits of the I-immediate field.
                    // The right shift type is encoded in bit 30. SLLI is a logical left shift (zeros are shifted into the lower bits);
                    // SRLI is a logical right shift (zeros are shifted into the upper bits);
                    // and SRAI is an arithmetic right shift (the original sign bit is copied into the vacated upper bits).
                    OP::SLLI => {
                        self.reg[rs1 as usize] = self.reg[rs1 as usize] << (imm_i & 0x1F);
                    }
                    OP::SRLI => {
                        self.reg[rs1 as usize] = self.reg[rs1 as usize] >> (imm_i & 0x1F);
                    }
                    OP::SRAI => {
                        self.reg[rs1 as usize] =
                            (self.reg[rs1 as usize] as i32 >> (imm_i & 0x1F)) as u32;
                    }
                }
            }
            OPS_TYPE::BRANCH(BRANCH) => match BRANCH {
                BRANCH::BEQ => todo!(),
                BRANCH::BNE => todo!(),
                BRANCH::BLT => todo!(),
                BRANCH::BGE => todo!(),
                BRANCH::BLTU => todo!(),
                BRANCH::BGEU => todo!(),
            },
        }
    }
}

pub fn extract_bits(s: u32, e: u32, inst: u32) -> u32 {
    inst >> e & (1 << (s - e + 1)) - 1
}

pub fn sign_extend(n: u32, bit: usize) -> u32 {
    if n >> (bit - 1) == 1 {
        -(((1 << bit) - n) as i32) as u32
    } else {
        n
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
fn rv32ui() -> Result<()> {
    use glob::glob;
    use object::elf::*;
    use object::read::elf::*;
    use object::read::{SectionIndex, StringTable};
    use object::Endianness;
    use object::{Object, ObjectSection};
    use std::fs::{self, File};
    use std::io::Read;
    for entry in glob("riscv-tests/isa/rv32ui-v-add").expect("fail to read path") {
        match entry {
            Ok(path) => {
                if !path.extension().is_some() {
                    let mut file = File::open(&path).expect("unable to read file");
                    let metadata = fs::metadata(&path).expect("unable to read metadata");
                    let mut buf = vec![0; metadata.len() as usize];
                    file.read_to_end(&mut buf)
                        .expect("unable to read into buffer");
                    println!("{:?}", path.display());
                    let bin_data = fs::read(&path)?;
                    let mut rv32i = RV32I::new();
                    let elf_file = ElfFile32::<Endianness>::parse(&*bin_data)?;
                    /* for segment in elf_file.segments() {
                        println!("{:?}", segment);
                    } */
                    for section in elf_file.sections() {
                        if section.address() < 0x8000_0000 {
                            continue;
                        }
                        let addr = (section.address() - 0x8000_0000) as usize;
                        let v: Vec<&u8> = section.data().unwrap().iter().collect();

                        for (i, byte) in section.data().unwrap().iter().enumerate() {
                            rv32i.memory[i + addr] = *byte;
                        }
                    }
                    /* for i in (0..rv32i.memory.len() - 4).step_by(4){
                        let b1 = rv32i.memory[i];
                        let b2 = rv32i.memory[i+1];
                        let b3 = rv32i.memory[i+2];
                        let b4 = rv32i.memory[i+3];

                        rv32i.memory[i] = b4;
                        rv32i.memory[i+1] = b3;
                        rv32i.memory[i+2] = b2;
                        rv32i.memory[i+3] = b1;
                    } */

                    /* let mut rv32i = RV32I::new();
                    for (i, chunk) in buf.chunks(4).enumerate() {
                        //println!("{:?}", &chunk );
                        let bytes = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        rv32i.memory[i] = bytes;
                    }
                    // println!("{:?}", rv32i.memory);
                    */
                    rv32i.reg[PC] = (elf_file.entry() - 0x8000_0000) as u32;
                    let addr = rv32i.reg[PC] as usize;
                    // println!("{:?}", rv32i.memory);
                    let mut counter = 0;
                    println!("endian:::: {:?}", elf_file.endian());
                    println!("pc at :::: {:032b}", rv32i.reg[PC]);
                    let em = u32::from_be_bytes([
                        rv32i.memory[addr],
                        rv32i.memory[addr + 1],
                        rv32i.memory[addr + 2],
                        rv32i.memory[addr + 3],
                    ]);
                    println!(
                        "b1: {:08b}\nb2: {:08b}\nb3: {:08b}\nb4: {:08b}",
                        rv32i.memory[addr],
                        rv32i.memory[addr + 1],
                        rv32i.memory[addr + 2],
                        rv32i.memory[addr + 3]
                    );
                    println!("ep mem:::: {:032b}  {:08x}", em, em);
                    while rv32i.cycle() {
                        counter += 1;
                        println!("{counter} instruction tested");
                    }
                }
            }
            Err(_) => panic!(),
        }
    }
    Ok(())
}
