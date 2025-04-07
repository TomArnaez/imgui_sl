module span;

public struct span<T> {
    private T* data_;
    private uint size_;

    public __subscript(uint x) -> T {
        get { return data_[x]; }
        set { data_[x] = newValue; }
    }

    public property uint size {
        get { return size_; }
    }
};

public struct mdspan<T, let Dims : uint> {
    private T *data_;
    private uint[Dims] extents_;

    private uint convert_index_to_1d(uint[Dims] indices) {
        uint index = 0;
        uint multiplier = 1;
        for (uint i = Dims - 1; i != 0; --i) {
            index += indices[i] * multiplier;
            multiplier *= extents_[i];
        }
        return index + indices[0];
    }

    public __subscript(uint[Dims] indices) -> T {
        get { return data_[convert_index_to_1d(indices)]; }
        set { data_[convert_index_to_1d(indices)] = newValue; }
    }
};