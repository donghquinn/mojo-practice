from String import *
from Vector import DynamicVector


struct CsvTable:
    var inner_string: String
    var starts: DynamicVector[Int]
    var ends: DynamicVector[Int]
    var column_count: Int

    fn __init__(inout self, owned s: String):
        self.inner_string = s
        self.starts = DynamicVector[Int](10)
        self.ends = DynamicVector[Int](10)
        self.column_count = -1
        self.parse()

    @always_inline
    fn parse(inout self):
        let QUOTE = ord('"')
        let COMMA = ord(",")
        let LF = ord("\n")
        let CR = ord("\r")
        let length = len(self.inner_string.buffer)
        var offset = 0
        var in_double_quotes = False
        self.starts.push_back(offset)
        while offset < length:
            let c = self.inner_string.buffer[offset]
            if c == QUOTE:
                in_double_quotes = not in_double_quotes
                offset += 1
            elif not in_double_quotes and c == COMMA:
                self.ends.push_back(offset)
                offset += 1
                self.starts.push_back(offset)
            elif not in_double_quotes and c == LF and not in_double_quotes:
                self.ends.push_back(offset)
                if self.column_count == -1:
                    self.column_count = len(self.ends)
                offset += 1
                self.starts.push_back(offset)
            elif (
                not in_double_quotes
                and c == CR
                and length > offset + 1
                and self.inner_string.buffer[offset + 1] == LF
            ):
                self.ends.push_back(offset)
                if self.column_count == -1:
                    self.column_count = len(self.ends)
                offset += 2
                self.starts.push_back(offset)
            else:
                offset += 1

        self.ends.push_back(length)

    fn get(self, row: Int, column: Int) -> String:
        if column >= self.column_count:
            return ""
        let index = self.column_count * row + column
        if index >= len(self.ends):
            return ""
        return self.inner_string[self.starts[index] : self.ends[index]]
