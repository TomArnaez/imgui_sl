enum print_op : uint32_t {
	new_line = 0,
    uint16,
    uint32
};

uint32_t* global_printer_buffer;

// Allocate space for one or more print commands
uint32_t _allocate_print_words(uint32_t word_count) {
    uint32_t word_offset = 0;
    InterlockedAdd(global_printer_buffer[0], word_count, word_offset);

    return word_offset + 1;
}

uint32_t _begin_println(uint32_t word_count) {
    // the word_count passed in represents the number of words required for the println arg,
    // but won't include the terminating string
    //
    // Thus we will allocate an extra word to allow us to append a new line to the print command
    uint32_t word_offset = _allocate_print_words(word_count + 1);

    global_printer_buffer[word_offset + word_count] = uint32_t(print_op.new_line) << 16;

    return word_offset;
}

void println() {
    _begin_println(0);
}

interface IPrintable {
    uint32_t print_word_count();
    void write_print_words(uint32_t* print_buffer, uint32_t offset);
};

void println<T : IPrintable>(T value) {
    uint32_t word_count = value.print_word_count();
    uint32_t word_offset = _begin_println(word_count);
    value.write_print_words(global_printer_buffer, word_offset);
}

extension uint16_t : IPrintable {
    uint32_t print_word_count() { return 2; }

    void write_print_words(uint32_t * print_buffer, uint32_t offset) {
        print_buffer[offset++] = (uint32_t(print_op.uint16) << 16) | 1;
        print_buffer[offset] = this;
    }
}

extension uint32_t : IPrintable {
    uint32_t print_word_count() { return 2; }

    void write_print_words(uint32_t* print_buffer, uint32_t offset) {
        print_buffer[offset++] = (uint32_t(print_op.uint32) << 16) | 1;
        print_buffer[offset] = this;
    }
}