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
    size_t n,                   
    opt_tr_fun_t fun,
    opt_tr_grad_t grad,
    opt_tr_hess_vec_t hess_vec,
    void *userdata
);

// destructor
void tr_method_template_free(tr_method_template *solver);

#endif
