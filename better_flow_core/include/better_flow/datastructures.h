#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H


// Circular Array
template <class DType, size_t SZ, long long SPAN> class CircularArray final {
protected:
    DType *data;
    DType *head;
    size_t current_size, head_id;
    bool span_checked;
    DType *latest;

public:
    typedef DType value_type;

    CircularArray () : current_size(0), head_id(0), span_checked(true), latest(NULL) {
        this->data = new DType[SZ];
        this->head = this->data;
    }

    ~CircularArray () {
        delete [] this->data;
    }

    inline size_t size () {
        this->fix_span();
        return this->current_size; 
    }

    inline void push_back (DType &d) {
        this->span_checked = false;
        this->current_size += (this->current_size >= SZ) ? 0 : 1;

        this->head_id ++;
        this->head ++;
        if (this->head_id >= SZ) {
            this->head_id = 0;
            this->head = this->data;
        }

        *(this->head) = d;
        this->latest = this->head;    
    }

    inline void fix_span () {
        if (this->span_checked) return;
        this->span_checked = true;
        size_t tail_id = ((1 - int(this->current_size - this->head_id)) + SZ) % SZ;
        size_t removed_cnt = 0;

        while ((long long)(*(this->latest) - this->data[tail_id]) > SPAN) {
            removed_cnt ++;
            tail_id ++;
            if (tail_id >= SZ) tail_id = 0;
        }

        this->current_size -= removed_cnt;
    }

    inline DType& operator [] (size_t idx) {
        assert (idx < this->current_size);
        return this->data[((int(this->head_id) - int(idx)) + SZ) % SZ];
    }

    inline auto begin() {
        this->fix_span();
        return _CAiterator(this->head, this->head_id, this->data); 
    }

    inline auto end()   {
        this->fix_span();
        int shift = (this->current_size >= SZ) ? 1 : 0;
        size_t tail_id = ((shift - int(this->current_size - this->head_id)) + SZ) % SZ;
        return _CAiterator(&(this->data[tail_id]), tail_id, this->data);
    }


protected:
    class _CAiterator {
    friend class CircularArray;
    public:
        DType& operator *() { return *(this->ptr); }
        DType* operator->() { return this->ptr; }

        _CAiterator& operator ++() {
            if (this->ptr_id == 0) {
                this->ptr_id = SZ - 1;
                this->ptr = &(this->data[this->ptr_id]);
                return *this;
            }

            this->ptr_id --;
            this->ptr --;           
            return *this;
        }

        bool operator !=(const _CAiterator &other) const {
            return this->ptr != other.ptr;
        }

        bool operator ==(const _CAiterator &other) const {
            return this->ptr == other.ptr;
        }

    protected:
        _CAiterator(DType *ptr_, size_t ptr_id_, DType *data_) 
            : ptr_id(ptr_id_), ptr(ptr_), data(data_) {}

    private:
        size_t ptr_id;
        DType *ptr;
        DType *data;
    };
};


// A simple linear event cloud with no structure
template <class DType> class LinearEventCloudTemplate {
protected:
    std::vector<DType> data;

public:
    int x_min, y_min, x_max, y_max;

public:
    LinearEventCloudTemplate () : x_min(INT_MAX), y_min(INT_MAX), x_max(INT_MIN), y_max(INT_MIN) {}

    LinearEventCloudTemplate (std::vector<DType> &data_)
        : x_min(INT_MAX), y_min(INT_MAX), x_max(INT_MIN), y_max(INT_MIN) {
        this->data.reserve(data_.size());
        for (auto &d : data_)
            this->push_back(d);
    }

    LinearEventCloudTemplate (std::vector<LinearEventCloudTemplate<DType>> &vec_data_)
        : x_min(INT_MAX), y_min(INT_MAX), x_max(INT_MIN), y_max(INT_MIN) {
        for (auto &data_ : vec_data_)
            for (auto &d : data_)
                this->push_back(d);
    }

    inline void push_back (DType d) {
        if ((int)d.get_x() > this->x_max) this->x_max = d.get_x();
        if ((int)d.get_y() > this->y_max) this->y_max = d.get_y();
        if ((int)d.get_x() < this->x_min) this->x_min = d.get_x();
        if ((int)d.get_y() < this->y_min) this->y_min = d.get_y();

        this->data.push_back(d);
    }

    inline DType& operator [] (size_t idx) {
        assert (idx < this->size());
        return this->data[idx];
    }

    inline size_t size () {
        return this->data.size(); 
    }

    inline auto begin() {
        return this->data.begin();
    }

    inline auto end()   {
        return this->data.end();
    }
};


// Same interface as LinearEventCloudTemplate, but storing pointers
template <class DType> class LinearEventPtrsTemplate {
protected:
    std::vector<DType*> data;

public:
    int x_min, y_min, x_max, y_max;

public:
    LinearEventPtrsTemplate () : x_min(INT_MAX), y_min(INT_MAX), x_max(INT_MIN), y_max(INT_MIN) {}

    LinearEventPtrsTemplate (std::vector<DType> &data_)
        : x_min(INT_MAX), y_min(INT_MAX), x_max(INT_MIN), y_max(INT_MIN) {
        this->data.reserve(data_.size());
        for (auto &d : data_)
            this->push_back(&d);
    }

    LinearEventPtrsTemplate (std::vector<LinearEventCloudTemplate<DType>> &vec_data_)
        : x_min(INT_MAX), y_min(INT_MAX), x_max(INT_MIN), y_max(INT_MIN) {
        for (auto &data_ : vec_data_)
            for (auto &d : data_)
                this->push_back(&d);
    }

    LinearEventPtrsTemplate (std::vector<LinearEventPtrsTemplate<DType>> &vec_data_)
        : x_min(INT_MAX), y_min(INT_MAX), x_max(INT_MIN), y_max(INT_MIN) {
        for (auto &data_ : vec_data_)
            for (auto &d : data_)
                this->push_back(d);
    }

    inline void push_back (DType *d) {
        if ((int)d->get_x() > this->x_max) this->x_max = d->get_x();
        if ((int)d->get_y() > this->y_max) this->y_max = d->get_y();
        if ((int)d->get_x() < this->x_min) this->x_min = d->get_x();
        if ((int)d->get_y() < this->y_min) this->y_min = d->get_y();

        this->data.push_back(d);
    }

    inline void push_back (DType &d) {
        this->push_back(&d);
    }

    inline DType& operator [] (size_t idx) {
        assert (idx < this->size());
        return *(this->data[idx]);
    }

    inline size_t size () {
        return this->data.size(); 
    }

    inline auto begin() {
        return _LEPiterator(this->data.begin());
    }

    inline auto end()   {
        return _LEPiterator(this->data.end());
    }

protected:
    class _LEPiterator {
    friend class LinearEventPtrsTemplate;
    public:
        DType& operator *() { return **(this->iter); }
        DType* operator->() { return *(this->iter); }

        _LEPiterator& operator ++() {
            this->iter++;
            return *this;
        }

        bool operator !=(const _LEPiterator &other) const {
            return this->iter != other.iter;
        }

        bool operator ==(const _LEPiterator &other) const {
            return this->iter == other.iter;
        }

    protected:
        _LEPiterator(typename std::vector<DType*>::iterator it_) : iter(it_) {}

    private:
        typename std::vector<DType*>::iterator iter;
    };
};


// Event Cloud
template <class DType, size_t SX, size_t SY> class EventCloudTemplate final {
protected:
    DType *data;

public:
    typedef DType value_type;

    EventCloudTemplate () {
        this->data = new DType[SX * SY];
    }

    ~EventCloudTemplate () {
        delete [] this->data;
    }

    inline void push_back (typename DType::value_type &d) {
        this->data[this->to_linear(d.get_x(), d.get_y())].push_back(d);
    }

    inline void push_back (typename DType::value_type &d, int x, int y) {
        this->data[this->to_linear(x, y)].push_back(d);
    }

    inline void push_back (DType &d, int x, int y) {
        this->data[this->to_linear(x, y)] = d;
    }

    inline void push_back (typename DType::value_type d) {
        this->data[this->to_linear(d.get_x(), d.get_y())].push_back(d);
    }

    inline void push_back (typename DType::value_type d, int x, int y) {
        this->data[this->to_linear(x, y)].push_back(d);
    }

    inline void push_back (DType d, int x, int y) {
        this->data[this->to_linear(x, y)] = d;
    }

    inline DType& get_col (size_t x, size_t y) {
        return this->data[this->to_linear(x, y)];
    }

    inline auto begin() {
        return _ECiterator(&(this->data[0]), 0, this->data); 
    }

    inline auto end()   {
        return _ECiterator(&(this->data[SX * SY - 1]), SX * SY - 1, this->data); 
    }

    inline DType& operator [] (size_t idx) {
        assert (idx < this->col_cnt());
        return this->data[idx];
    }

    inline size_t col_cnt () const {return SX * SY; }
    inline size_t get_sx ()  const {return SX; }
    inline size_t get_sy ()  const {return SY; }

protected:
    inline size_t to_linear(size_t x, size_t y) const {
        assert(x < SX && y < SY);
        return y * SX + x;
    }

    class _ECiterator {
    friend class EventCloudTemplate;
    public:
        typename DType::value_type& operator *() { return *(this->left_it); }
        typename DType::value_type* operator->() { return this->left_it; }

        _ECiterator& operator ++() {
            if (this->left_it == this->rght_it) {
                if (this->ptr_id < SX * SY - 1) ++ this->ptr_id;
                this->reach_next_nonempty();
                return *this;
            }

            ++ this->left_it;
            return *this;
        }

        /*
        _ECiterator& operator ++() {
            this->ptr ++;
            this->left_it ++;
            return *this;
        }
        */

        bool operator !=(const _ECiterator &other) const {
            return ((this->ptr_id != other.ptr_id) || (this->left_it != other.rght_it));
        }

        bool operator ==(const _ECiterator &other) const {
            return ((this->ptr_id == other.ptr_id) && (this->left_it == other.rght_it));
        }

    protected:
        _ECiterator(DType *ptr_, size_t ptr_id_, DType *data_) 
            : ptr_id(ptr_id_), ptr(ptr_), data(data_), left_it(ptr_->begin()), rght_it(ptr_->end()) {
            this->reach_next_nonempty();
        }

        /*
        _ECiterator(DType *ptr_, size_t ptr_id_, DType *data_) 
            : ptr_id(ptr_id_), ptr(ptr_), data(data_) {
            this->left_it = ptr_id_;
            this->rght_it = ptr_id_;
        }
        */

        void reach_next_nonempty() { 
            while ((this->ptr_id < SX * SY - 1) && (this->data[this->ptr_id].size() == 0)) {
                this->ptr_id ++;
            }

            this->ptr = &(this->data[this->ptr_id]);
            this->left_it = this->ptr->begin();
            this->rght_it = this->ptr->end();
        }

    private:
        size_t ptr_id;
        DType *ptr;
        DType *data;

        decltype(ptr->begin()) left_it, rght_it;
    };
};



#endif // DATASTRUCTURES_H
