struct span<T> {
    private T* data_;
    private uint size_;

    __subscript(uint x) -> T {
        get { return data_[x]; }
        set { data_[x] = newValue; }
    }

    property uint size {
        get { return size_; }
    }
};