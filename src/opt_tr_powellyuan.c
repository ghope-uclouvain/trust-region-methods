#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/opt_tr_powellyuan.h"
#include "../include/opt_tr.h"

// linear algebra helpers
static double dot(const double *a, const double *b, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        s += a[i] * b[i];
    return s;
}

// L2, euclidean norm
static double nrm2(const double *x, size_t n)
{
    return sqrt(fmax(0.0, dot(x, x, n)));
}

// y = ax + y
static void axpy(double alpha, const double *x, double *y, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}

// applies scalar
static void scal(double alpha, double *x, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        x[i] *= alpha;
}

// copies two vectors
static void copy(const double *src, double *dst, size_t n)
{
    if (src == dst) return;
    memcpy(dst, src, n * sizeof(double));
}

// fills vector with a singular value
static void fill(double *x, size_t n, double v)
{
    for (size_t i = 0; i < n; ++i)
        x[i] = v;
}


// build jacobian column by column 
// associates with eq. (1.4)
static void build_jacobian_at(opt_tr *tr, const double *x, double *J_out)
{
    // if missing anything essential
    if (!tr || !tr->vptr || !tr->vptr->constr_jv || !x || !J_out){
        return;
    }

    size_t n = tr->n;
    size_t m = tr->m;

    double *v  = (double*)calloc(n, sizeof(double));
    double *Jv = (double*)calloc(m, sizeof(double));

    if (!v || !Jv) {
        free(v); free(Jv);
        return;
    }

    // solves Jv = Je_i
    for (size_t j = 0; j < n; ++j) {
        fill(v, n, 0.0);
        v[j] = 1.0;
        tr->vptr->constr_jv(x, v, Jv);
        for (size_t i = 0; i < m; ++i)
            J_out[i*n + j] = Jv[i];
    }

    free(v);
    free(Jv);
}

// gaussian elimination with partial pivoting
// just a linear solver
 int gauss_pp_solve(double *A, double *b, size_t n)
{
    if (!A || !b) return -1;

    // select the pivot with the largest abs. value
    for (size_t k = 0; k < n; ++k) {
        // pivot selection 
        size_t piv  = k;
        double maxv = fabs(A[k*n + k]);
        for (size_t i = k + 1; i < n; ++i) {
            double v = fabs(A[i*n + k]);
            if (v > maxv) { maxv = v; piv = i; }
        }

        if (maxv < 1e-16)
            return -1; // singular

        // if pivot is different than k
        if (piv != k) {
            for (size_t j = k; j < n; ++j) {
                double tmp      = A[k*n + j];
                A[k*n + j]      = A[piv*n + j];
                A[piv*n + j]    = tmp;
            }
            double tb = b[k];
            b[k]      = b[piv];
            b[piv]    = tb;
        }

        double diag = A[k*n + k];
        for (size_t i = k + 1; i < n; ++i) {
            double f = A[i*n + k] / diag;
            for (size_t j = k; j < n; ++j)
                A[i*n + j] -= f * A[k*n + j];
            b[i] -= f * b[k];
        }
    }

    // back substitution
    for (ptrdiff_t ii = (ptrdiff_t)n - 1; ii >= 0; --ii) {
        size_t i = (size_t)ii;
        double s = b[i];
        for (size_t j = i + 1; j < n; ++j)
            s -= A[i*n + j] * b[j];
        b[i] = s / A[i*n + i];
    }

    return 0;
}

// find lambda from min || g_k - A_k lambda ||
void solve_min_norm_lambda(const double *A, const double *g,
                           size_t n, size_t m, double *lambda_out)
{
    if (!A || !g || !lambda_out || m == 0)
        return;

    double *AAT = (double*)calloc(m*m, sizeof(double));
    double *Ag  = (double*)calloc(m,   sizeof(double));
    if (!AAT || !Ag) {
        free(AAT); free(Ag);
        return;
    }

    // A A^T and A g (A is m×n row-major) 
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += A[i*n + k] * A[j*n + k];
            AAT[i*m + j] = s;
        }
        double t = 0.0;
        for (size_t k = 0; k < n; ++k)
            t += A[i*n + k] * g[k];
        Ag[i] = t;
    }

    memcpy(lambda_out, Ag, m * sizeof(double));
    gauss_pp_solve(AAT, lambda_out, m);

    free(AAT);
    free(Ag);
}


//   projection of v onto nullspace of A^T (A = J, m×n)
//   part of solving the CDT subproblem
void project_nullspace(const double *J, size_t m, size_t n,
                              const double *v, double *out)
{
    if (!J || m == 0) {
        copy(v, out, n);
        return;
    }

    double *Jv  = (double*)calloc(m,   sizeof(double));
    double *JJt = (double*)calloc(m*m, sizeof(double));
    if (!Jv || !JJt) {
        free(Jv); free(JJt);
        copy(v, out, n);
        return;
    }

    // Jv = J v 
    for (size_t i = 0; i < m; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < n; ++j)
            s += J[i*n + j] * v[j];
        Jv[i] = s;
    }

    // JJt = J J^T
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += J[i*n + k] * J[j*n + k];
            JJt[i*m + j] = s;
        }
    }

    // (JJ^T) y = J v
    if (gauss_pp_solve(JJt, Jv, m) != 0) {
        copy(v, out, n);
        free(Jv); free(JJt);
        return;
    }

    // out = v − J^T y
    copy(v, out, n);
    for (size_t j = 0; j < n; ++j) {
        double corr = 0.0;
        for (size_t i = 0; i < m; ++i)
            corr += J[i*n + j] * Jv[i];
        out[j] -= corr;
    }

    free(Jv);
    free(JJt);
}

// solve linear feasibility correction: J d_c = -c_k
// part of cdt subproblem
int solve_linear_constraint(opt_tr *tr, const double *J, double *d_c_out)
{
    if (!tr || !d_c_out) return -1;

    size_t m = tr->m;
    size_t n = tr->n;

    if (m == 0 || !J) {
        fill(d_c_out, n, 0.0);
        return 0;
    }

    double *JJt = (double*)calloc(m*m, sizeof(double));
    double *rhs = (double*)calloc(m,   sizeof(double));
    if (!JJt || !rhs) {
        free(JJt); free(rhs);
        return -1;
    }

    /* JJᵀ and rhs = −c */
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (size_t k = 0; k < n; ++k)
                s += J[i*n + k] * J[j*n + k];
            JJt[i*m + j] = s;
        }
        rhs[i] = -tr->c_k[i];
    }

    int rc = gauss_pp_solve(JJt, rhs, m);
    if (rc != 0) {
        /* fallback: no correction */
        fill(d_c_out, n, 0.0);
        free(JJt); free(rhs);
        return 0;
    }

    /* d_c = Jᵀ y */
    for (size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (size_t i = 0; i < m; ++i)
            s += J[i*n + j] * rhs[i];
        d_c_out[j] = s;
    }

    free(JJt);
    free(rhs);
    return 0;
}

// truncated CG for the nullspace step, part of the CDT subproblem
static int tCG_nullspace(opt_tr *tr, const double *J,
                         double *s_out,
                         int maxit, double tol)
{
    if (!tr || !s_out) return -1;

    size_t n = tr->n;
    size_t m = tr->m;

    double *s   = (double*)calloc(n, sizeof(double));
    double *r   = (double*)calloc(n, sizeof(double));
    double *p   = (double*)calloc(n, sizeof(double));
    double *Hp  = (double*)calloc(n, sizeof(double));
    double *tmp = (double*)calloc(n, sizeof(double));

    if (!s || !r || !p || !Hp || !tmp) {
        free(s); free(r); free(p); free(Hp); free(tmp);
        return -1;
    }

    /* r = −g projected onto nullspace */
    for (size_t i = 0; i < n; ++i)
        r[i] = -tr->gradient[i];

    project_nullspace(J, m, n, r, r);
    copy(r, p, n);

    double rTr    = dot(r, r, n);
    double radius = tr->radius;
    int    it_max = (maxit > 0 ? maxit : 1000);

    if (nrm2(r, n) < tol) {
        fill(s_out, n, 0.0);
        free(s); free(r); free(p); free(Hp); free(tmp);
        return 0;
    }

    for (int it = 0; it < it_max; ++it) {
        if (tr->vptr && tr->vptr->hess_vec)
            tr->vptr->hess_vec(tr->x_k, p, Hp);
        else
            copy(p, Hp, n);

        project_nullspace(J, m, n, Hp, Hp);

        double pHp = dot(p, Hp, n);
        if (fabs(pHp) < 1e-16)
            break;

        double alpha = rTr / pHp;

        /* tentative step */
        copy(s, tmp, n);
        axpy(alpha, p, tmp, n);
        double n_new = nrm2(tmp, n);

        if (n_new > radius + 1e-14) {
            /* boundary hit: solve quadratic for t in [0,alpha] */
            double sTp = dot(s, p, n);
            double pTp = dot(p, p, n);
            double sTs = dot(s, s, n);
            double disc = sTp*sTp + pTp*(radius*radius - sTs);
            double t = (disc > 0.0)
                     ? (-sTp + sqrt(disc)) / pTp
                     : 0.0;
            axpy(t, p, s, n);
            break;
        }

        axpy(alpha, p, s, n);       /* accept step */
        axpy(-alpha, Hp, r, n);     /* update residual */

        double rTr_new = dot(r, r, n);
        if (nrm2(r, n) < tol)
            break;

        double beta = rTr_new / rTr;
        for (size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * p[i];

        rTr = rTr_new;
    }

    copy(s, s_out, n);

    free(s); free(r); free(p); free(Hp); free(tmp);
    return 0;
}

// powell-yuan update radius rule 
 int py_update_radius(opt_tr *tr, double rho){
    tr_method_powellyuan *ctx = POWELLYUAN_CAST(tr);

    // thresholds from ctx, but give powell-yuan defaults
    double b1 = (ctx->b1 > 0.0 && ctx->b1 < 1.0) ? ctx->b1 : 0.9;
    double b2 = (ctx->b2 > 0.0 && ctx->b2 < b1)   ? ctx->b2 : 0.1;

    /* step length */
    double *s = (double*)calloc(tr->n, sizeof(double));
    if (!s) return -1;
    for (size_t i = 0; i < tr->n; ++i)
        s[i] = tr->test_x[i] - tr->x_k[i];
    double sn = nrm2(s, tr->n);
    free(s);

    if (rho >= b1) {
        // good step, possibly enlarge radius
        double new_r = fmax(tr->radius, 4.0 * sn);
        tr->radius   = fmin(new_r, tr->config.radius_max);
    } else if (rho < b2) {
        // poor step, shrink radius
        double new_r = tr->radius / 4.0;
        if (sn > 0.0)
            new_r = fmin(new_r, (sn*sn) / 2.0);
        tr->radius = fmax(new_r, 1e-12);
    } // else: keep radius

    return 0;
}

 int py_accept_step(opt_tr *tr, double rho)
{
    //accept if rho > 0
    if (rho > 0.0) {
        memcpy(tr->x_k, tr->test_x, tr->n * sizeof(double));

        if (tr->vptr && tr->vptr->grad)
            tr->vptr->grad(tr->x_k, tr->gradient);
        if (tr->m > 0 && tr->vptr && tr->vptr->constraints)
            tr->vptr->constraints(tr->x_k, tr->c_k);

        return 1;
    }
    return 0;
}


// have not implemented a quasi newton function yet
 int py_qn_update(opt_tr *tr)
{
    (void)tr;
    return 0; 
}

// vtable wrappers
static int py_update_radius_cb(opt_tr *tr, double ratio)
{
    return py_update_radius(tr, ratio);
}
static int py_accept_step_cb(opt_tr *tr, double ratio)
{
    return py_accept_step(tr, ratio);
}
static int py_qn_update_cb(opt_tr *tr)
{
    return py_qn_update(tr);
}


int powellyuan_step_cb(opt_tr *tr, double *expected_improvement)
{
    tr_method_powellyuan *ctx = POWELLYUAN_CAST(tr);
    size_t n = tr->n;
    size_t m = tr->m;

    if (ctx->max_cg <= 0) ctx->max_cg = 200;
    if (ctx->cg_tol <= 0) ctx->cg_tol = 1e-8;
    if (ctx->mu <= 0.0)   ctx->mu     = 1.0;

    // Ensure grad and constraints at x_k are up-to-date 
    if (tr->vptr->grad)
        tr->vptr->grad(tr->x_k, tr->gradient);
    if (m > 0 && tr->vptr->constraints)
        tr->vptr->constraints(tr->x_k, tr->c_k);

    // 1. Build Jacobian J_k and lambda_k at x_k */
    if (m > 0) {
        build_jacobian_at(tr, tr->x_k, ctx->A);
        solve_min_norm_lambda(ctx->A, tr->gradient, n, m, ctx->lambda_k);
    }

    //2. CDT step
    double Delta = tr->radius;

    double *d_c = (double*)calloc(n, sizeof(double));
    double *s_n = (double*)calloc(n, sizeof(double));
    double *s   = (double*)calloc(n, sizeof(double));
    if (!d_c || !s_n || !s) {
        free(d_c); free(s_n); free(s);
        return -1;
    }

    // Feasibility correction J d_c = −c
    if (m > 0)
        solve_linear_constraint(tr, ctx->A, d_c);
    else
        fill(d_c, n, 0.0);

    double norm_dc = nrm2(d_c, n);

    if (norm_dc >= Delta) {
        // clip d_c to trust-region boundary
        if (norm_dc > 0.0)
            scal(Delta / norm_dc, d_c, n);
        copy(d_c, s, n);
        fill(s_n, n, 0.0);
    } else {
        // remaining radius for nullspace step
        double rem = sqrt(Delta*Delta - norm_dc*norm_dc);

        if (rem > 0.0) {
            double saved = tr->radius;
            tr->radius = rem;
            tCG_nullspace(tr, (m > 0 ? ctx->A : NULL),
                          s_n, ctx->max_cg, ctx->cg_tol);
            tr->radius = saved;
        }

        for (size_t i = 0; i < n; ++i)
            s[i] = d_c[i] + s_n[i];

        // safeguard against numerical drift
        double ns = nrm2(s, n);
        if (ns > Delta)
            scal(Delta / ns, s, n);
    }

    // trial point
    for (size_t i = 0; i < n; ++i)
        tr->test_x[i] = tr->x_k[i] + s[i];

    // 3. merit function and model D_k

    double f_k = 0.0, f_trial = 0.0;
    tr->vptr->fun(tr->x_k,     &f_k);
    tr->vptr->fun(tr->test_x,  &f_trial);

    double *c_trial = (m > 0 ? ctx->c_trial : NULL);
    if (m > 0 && tr->vptr->constraints)
        tr->vptr->constraints(tr->test_x, c_trial);

    double psi_k     = f_k;
    double psi_trial = f_trial;

    if (m > 0) {
        double lamTc_k = 0.0;
        double norm_ck2 = 0.0;
        for (size_t i = 0; i < m; ++i) {
            lamTc_k   += ctx->lambda_k[i] * tr->c_k[i];
            norm_ck2  += tr->c_k[i]       * tr->c_k[i];
        }

        psi_k = f_k - lamTc_k + ctx->mu * norm_ck2;

        double lamTc_trial = 0.0;
        double norm_ct2    = 0.0;
        for (size_t i = 0; i < m; ++i) {
            lamTc_trial += ctx->lambda_k[i] * c_trial[i];
            norm_ct2    += c_trial[i]       * c_trial[i];
        }

        psi_trial = f_trial - lamTc_trial + ctx->mu * norm_ct2;
    }


    double *Js = NULL;
    double *Bs = (double*)calloc(n, sizeof(double));
    double *g_minus_Jtlam = (double*)calloc(n, sizeof(double));
    if (m > 0)
        Js = (double*)calloc(m, sizeof(double));

    if (tr->vptr->hess_vec)
        tr->vptr->hess_vec(tr->x_k, s, Bs);
    else
        copy(s, Bs, n);

    if (m > 0) {
        double *Jtlam = (double*)calloc(n, sizeof(double));
        for (size_t j = 0; j < n; ++j) {
            double v = 0.0;
            for (size_t i = 0; i < m; ++i)
                v += ctx->A[i*n + j] * ctx->lambda_k[i];
            Jtlam[j] = v;
        }
        for (size_t j = 0; j < n; ++j)
            g_minus_Jtlam[j] = tr->gradient[j] - Jtlam[j];
        free(Jtlam);
    } else {
        copy(tr->gradient, g_minus_Jtlam, n);
    }

    double pen = 0.0;
    if (m > 0) {
        for (size_t i = 0; i < m; ++i) {
            double v = 0.0;
            for (size_t j = 0; j < n; ++j)
                v += ctx->A[i*n + j] * s[j];
            Js[i] = v;
        }

        double norm_ck2 = 0.0;
        double norm_cl2 = 0.0;
        for (size_t i = 0; i < m; ++i) {
            norm_ck2 += tr->c_k[i] * tr->c_k[i];
            double cl = tr->c_k[i] + Js[i];
            norm_cl2 += cl * cl;
        }
        pen = ctx->mu * (norm_cl2 - norm_ck2);
    }

    double term_lin  = dot(g_minus_Jtlam, s, n);
    double term_quad = 0.5 * dot(s, Bs, n);
    double Dk        = term_lin + term_quad + pen; 

    double predicted_decrease = 0.0;
    if (Dk < 0.0)
        predicted_decrease = -Dk;
    else
        predicted_decrease = 1e-12; /* safeguard */

    double actual_decrease = psi_k - psi_trial;
    double rho = 0.0;
    if (predicted_decrease > 0.0)
        rho = actual_decrease / predicted_decrease;

    if (expected_improvement)
        *expected_improvement = predicted_decrease;

    // 4. Use p to update radius & accept/reject via vtable

    if (tr->vptr->update_radius)
        tr->vptr->update_radius(tr, rho);
    if (tr->vptr->accept_step)
        tr->vptr->accept_step(tr, rho);
    if (tr->vptr->qn_update)
        tr->vptr->qn_update(tr);

    free(d_c);
    free(s_n);
    free(s);
    free(Js);
    free(Bs);
    free(g_minus_Jtlam);

    return 0;
}

int tr_method_powellyuan_init(
    tr_method_powellyuan *solver,
    size_t m,
    size_t n,
    double *x_k,
    double *test_x,
    double *gradient,
    double radius,
    double *c_k,
    opt_tr_options_t config,
    opt_tr_fun_t fun,
    opt_tr_grad_t grad,
    opt_tr_hess_vec_t hess_vec,
    opt_tr_constraints_t cons,
    opt_tr_constr_jv_t constr_jv)
{
    if (!solver) return -1;

    static opt_tr_vtable_t vtbl;

    vtbl.fun         = fun;
    vtbl.grad        = grad;
    vtbl.hess_vec    = hess_vec;
    vtbl.constraints = cons;
    vtbl.constr_jv   = constr_jv;

    vtbl.step          = powellyuan_step_cb;
    vtbl.update_radius = py_update_radius_cb;
    vtbl.accept_step   = py_accept_step_cb;
    vtbl.qn_update     = py_qn_update_cb;

    if (opt_tr_init(&solver->base, m, n,
                    x_k, test_x, gradient,
                    radius, c_k, config, &vtbl) != 0)
        return -1;

    solver->mu     = 1.0;
    solver->b1     = 0.9;  
    solver->b2     = 0.10; 
    solver->max_cg = 200;
    solver->cg_tol = 1e-8;

    solver->lambda_k = (m ? (double*)calloc(m,   sizeof(double)) : NULL);
    solver->c_trial  = (m ? (double*)calloc(m,   sizeof(double)) : NULL);
    solver->A        = (m ? (double*)calloc(m*n, sizeof(double)) : NULL);
    solver->Hs       = (n ? (double*)calloc(n,   sizeof(double)) : NULL);

    if ((m && (!solver->lambda_k || !solver->c_trial || !solver->A)) ||
        (n && !solver->Hs)) {
        tr_method_powellyuan_free(solver);
        return -1;
    }

    return 0;
}

void tr_method_powellyuan_free(tr_method_powellyuan *solver)
{
    if (!solver) return;
    free(solver->lambda_k); solver->lambda_k = NULL;
    free(solver->c_trial ); solver->c_trial  = NULL;
    free(solver->A       ); solver->A        = NULL;
    free(solver->Hs      ); solver->Hs       = NULL;
}
