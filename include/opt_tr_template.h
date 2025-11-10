#ifndef OPT_TR_TEMPLATE
#define OPT_TR_TEMPLATE

#include "opt_tr.h"

typedef struct {
    opt_tr base;   // 'inherits' from opt_tr.h
    // add any method specific data that you need here
} tr_method_template;

// constructor
int tr_method_template_init(
    tr_method_template *solver,
    size_t n,                    // dimension of the problem
    long double x_k,             // initial value of x
    opt_tr_fun_t fun,            // objective function
    opt_tr_constraints_t cons   // constraint function
);

// destructor
void tr_method_template_free(tr_method_template *solver);

#endif
