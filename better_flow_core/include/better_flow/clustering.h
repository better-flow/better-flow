#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <list>

#include <better_flow/common.h>
#include <better_flow/event.h>


class Cluster {
protected:
    // Unique id generator
    static unsigned int cnt;
    unsigned int id;
    
public:
    Cluster ();

    inline unsigned int get_id () const {return this->id; }
    void add (Event &e);

    bool operator==(const Cluster &other) const {
        return this->id == other.id;
    }

    Cluster& operator+=(Cluster& rhs);
};


#endif // CLUSTERING_H
