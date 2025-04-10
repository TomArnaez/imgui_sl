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

public struct mdspan<T, let dims : uint> {
    private T *data_;
    private uint[dims] extents_;

    private uint convert_index_to_1d(uint[dims] indices) {
        uint index = 0;
        uint multiplier = 1;
        for (int i = (int)dims - 1; i >= 0; --i) {
            index += indices[i] * multiplier;
            multiplier *= extents_[i];
        }
        return index;
    }

    public __subscript(uint[dims] indices) -> T {
        get { return data_[convert_index_to_1d(indices)]; }
        set { data_[convert_index_to_1d(indices)] = newValue; }
    }

    public property uint[dims] extents {
        get { return extents_; }
    }
};