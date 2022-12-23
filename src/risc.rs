#![allow(non_snake_case)]

pub enum RV32I_OPCODE {
    LUI    = 0b0110111,
    AUIPC  = 0b0010111,
    JAL    = 0b1101111,
    JALR   = 0b1100111,
    FENCE  = 0b0001111,
    /*
    BEQ    = 0b1100011,
    BNE    = 0b1100011,
    BLT    = 0b1100011,
    BGE    = 0b1100011,
    BLTU   = 0b1100011,
    BGEU   = 0b1100011,

    LB     = 0b0000011,
    LH     = 0b0000011,
    LW     = 0b0000011,
    LBU    = 0b0000011,
    LHU    = 0b0000011,

    SB     = 0b0100011,
    SH     = 0b0100011,
    SW     = 0b0100011,
    */
    OP     = 0b0100011,

    /*
    ADDI   = 0b0010011,
    SLTI   = 0b0010011,
    SLTIU  = 0b0010011,
    XORI   = 0b0010011,
    ORI    = 0b0010011,
    ANDI   = 0b0010011,
    SLLI   = 0b0010011,
    SRLI   = 0b0010011,
    SRAI   = 0b0010011,
    ADD    = 0b0110011,
    SUB    = 0b0110011,
    SLL    = 0b0110011,
    SLT    = 0b0110011,
    SLTU   = 0b0110011,
    XOR    = 0b0110011,
    SRL    = 0b0110011,
    SRA    = 0b0110011,
    OR     = 0b0110011,
    AND    = 0b0110011,
    */
    SHIFT  = 0b0110011,

    /*
    ECALL  = 0b1110011,
    EBREAK = 0b1110011,
    */
    SYSTEM = 0b1110011,
}

pub struct RV32I {
    pub(crate) memory: [u32; 10000],
    pub(crate) PC: usize,
}

impl RV32I {
    pub fn new() -> Self {
        Self {
            memory: [0; 10000],
            PC: 0,
        }
    }

    pub fn opcode(&self) -> u8 {
        // opcode = 0-6 digits
        return (self.memory[self.PC] & 0x7F ) as u8
    }
}

#[test]
fn test() {
}
