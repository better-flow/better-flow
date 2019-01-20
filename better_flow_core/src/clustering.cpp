#include <better_flow/clustering.h>
#include <better_flow/event_file.h>
#include <better_flow/object_model.h>
#include <better_flow/accel_lib.h>


// Cluster class
unsigned int Cluster::cnt = 0;


Cluster::Cluster () {
    cnt++;
    this->id = cnt;
}


void Cluster::add (Event &e) { 
    e.cl = this;
}


Cluster& Cluster::operator+=(Cluster& rhs) {
    rhs.id = this->id;
    return *this;
}
