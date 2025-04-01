struct span<T> {
    T* data;
    uint size;

    __subscript(uint x) -> T {
        get { return data[x]; }
        set { data[x] = newValue; }
    }
};