/*
    Copyright (C) 2016 Tomas Flouri, Diego Darriba, Alexey Kozlov

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact: Tomas Flouri <Tomas.Flouri@h-its.org>,
    Exelixis Lab, Heidelberg Instutute for Theoretical Studies
    Schloss-Wolfsbrunnenweg 35, D-69118 Heidelberg, Germany
*/
#include <limits.h>
#include "pll.h"

inline double reduce_add_pd(const __m512d zmm) {
  __m256d low = _mm512_castpd512_pd256(zmm);
  __m256d high = _mm512_extractf64x4_pd(zmm, 1);

  __m256d a = _mm256_add_pd(low, high);
  __m256d t1 = _mm256_hadd_pd(a, a);
  __m128d t2 = _mm256_extractf128_pd(t1, 1);
  __m128d t3 = _mm_add_sd(_mm256_castpd256_pd128(t1), t2);
  return _mm_cvtsd_f64(t3);
}

#define COMPUTE_II_QCOL_HALF(q, offset) \
/* row 0 */ \
v_mat    = _mm512_load_pd(lm0 + (offset)); \
v_lterm0 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm0); \
v_mat    = _mm512_load_pd(rm0 + (offset)); \
v_rterm0 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm0); \
\
/* row 1 */ \
v_mat    = _mm512_load_pd(lm1 + (offset)); \
v_lterm1 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm1); \
v_mat    = _mm512_load_pd(rm1 + (offset)); \
v_rterm1 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm1); \
\
/* row 2 */ \
v_mat    = _mm512_load_pd(lm2 + (offset)); \
v_lterm2 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm2); \
v_mat    = _mm512_load_pd(rm2 + (offset)); \
v_rterm2 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm2); \
\
/* row 3 */ \
v_mat    = _mm512_load_pd(lm3 + (offset)); \
v_lterm3 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm3); \
v_mat    = _mm512_load_pd(rm3 + (offset)); \
v_rterm3 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm3);

#define COMPUTE_II_QCOL(q, offset) \
/* row 0 */ \
v_mat    = _mm512_load_pd(lm0 + (offset)); \
v_lterm0 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm0); \
v_mat    = _mm512_load_pd(rm0 + (offset)); \
v_rterm0 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm0); \
\
/* row 1 */ \
v_mat    = _mm512_load_pd(lm1 + (offset)); \
v_lterm1 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm1); \
v_mat    = _mm512_load_pd(rm1 + (offset)); \
v_rterm1 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm1); \
\
/* row 2 */ \
v_mat    = _mm512_load_pd(lm2 + (offset)); \
v_lterm2 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm2); \
v_mat    = _mm512_load_pd(rm2 + (offset)); \
v_rterm2 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm2); \
\
/* row 3 */ \
v_mat    = _mm512_load_pd(lm3 + (offset)); \
v_lterm3 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm3); \
v_mat    = _mm512_load_pd(rm3 + (offset)); \
v_rterm3 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm3); \
\
/* row 4 */ \
v_mat    = _mm512_load_pd(lm4 + (offset)); \
v_lterm4 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm4); \
v_mat    = _mm512_load_pd(rm4 + (offset)); \
v_rterm4 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm4); \
\
/* row 5 */ \
v_mat    = _mm512_load_pd(lm5 + (offset)); \
v_lterm5 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm5); \
v_mat    = _mm512_load_pd(rm5 + (offset)); \
v_rterm5 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm5); \
\
/* row 6 */ \
v_mat    = _mm512_load_pd(lm6 + (offset)); \
v_lterm6 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm6); \
v_mat    = _mm512_load_pd(rm6 + (offset)); \
v_rterm6 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm6); \
\
/* row 7 */ \
v_mat    = _mm512_load_pd(lm7 + (offset)); \
v_lterm7 = _mm512_fmadd_pd(v_mat, v_lclv[q], v_lterm7); \
v_mat    = _mm512_load_pd(rm7 + (offset)); \
v_rterm7 = _mm512_fmadd_pd(v_mat, v_rclv[q], v_rterm7);

#define COMPUTE_II_QROW_8(j) {                                                            \
/* point to the four rows of the eigenvecs matrix */                                      \
  const double *lm0 = ct_inv_eigenvecs;                                                   \
  const double *lm1 = lm0 + states_padded;                                                \
  const double *lm2 = lm1 + states_padded;                                                \
  const double *lm3 = lm2 + states_padded;                                                \
  const double *lm4 = lm3 + states_padded;                                                \
  const double *lm5 = lm4 + states_padded;                                                \
  const double *lm6 = lm5 + states_padded;                                                \
  const double *lm7 = lm6 + states_padded;                                                \
  ct_inv_eigenvecs += ELEM_PER_AVX515_REGISTER * states_padded;                           \
                                                                                          \
  /* point to the four rows of the inv_eigenvecs matrix */                                \
  const double *rm0 = c_eigenvecs;                                                        \
  const double *rm1 = rm0 + states_padded;                                                \
  const double *rm2 = rm1 + states_padded;                                                \
  const double *rm3 = rm2 + states_padded;                                                \
  const double *rm4 = rm3 + states_padded;                                                \
  const double *rm5 = rm4 + states_padded;                                                \
  const double *rm6 = rm5 + states_padded;                                                \
  const double *rm7 = rm6 + states_padded;                                                \
  c_eigenvecs += ELEM_PER_AVX515_REGISTER * states_padded;                                \
                                                                                          \
  __m512d v_lterm0 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm0 = _mm512_setzero_pd();                                                 \
  __m512d v_lterm1 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm1 = _mm512_setzero_pd();                                                 \
  __m512d v_lterm2 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm2 = _mm512_setzero_pd();                                                 \
  __m512d v_lterm3 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm3 = _mm512_setzero_pd();                                                 \
  __m512d v_lterm4 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm4 = _mm512_setzero_pd();                                                 \
  __m512d v_lterm5 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm5 = _mm512_setzero_pd();                                                 \
  __m512d v_lterm6 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm6 = _mm512_setzero_pd();                                                 \
  __m512d v_lterm7 = _mm512_setzero_pd();                                                 \
  __m512d v_rterm7 = _mm512_setzero_pd();                                                 \
                                                                                          \
  __m512d v_mat;                                                                          \
                                                                                          \
  /* iterate over quadruples of columns */                                                \
  COMPUTE_II_QCOL(0, 0);                                                                  \
  COMPUTE_II_QCOL(1, 8);                                                                  \
  COMPUTE_II_QCOL(2, 16);                                                                 \
                                                                                          \
  /* compute lefterm */                                                                   \
  __m512d xmm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_lterm0, v_lterm1),                    \
                               _mm512_unpacklo_pd(v_lterm0, v_lterm1));                   \
  __m512d xmm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_lterm2, v_lterm3),                    \
                               _mm512_unpacklo_pd(v_lterm2, v_lterm3));                   \
  __m512d xmm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_lterm4, v_lterm5),                    \
                               _mm512_unpacklo_pd(v_lterm4, v_lterm5));                   \
  __m512d xmm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_lterm6, v_lterm7),                    \
                               _mm512_unpacklo_pd(v_lterm6, v_lterm7));                   \
                                                                                          \
  __m512d ymm0 = _mm512_add_pd(_mm512_permutex2var_pd(xmm0, permute_mask, xmm2),          \
                               _mm512_mask_blend_pd(0xF0, xmm0, xmm2));                   \
                                                                                          \
  __m512d ymm1 = _mm512_add_pd(_mm512_permutex2var_pd(xmm1, permute_mask, xmm3),          \
                               _mm512_mask_blend_pd(0xF0, xmm1, xmm3));                   \
                                                                                          \
  __m512d v_lefterm_sum = _mm512_add_pd(_mm512_permutex2var_pd(ymm0,                      \
                                                               permute_mask_final_stage,  \
                                                               ymm1),                     \
                                        _mm512_mask_blend_pd(0xCC, ymm0, ymm1));          \
                                                                                          \
  /* compute righterm */                                                                  \
  xmm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_rterm0, v_rterm1),                            \
                       _mm512_unpacklo_pd(v_rterm0, v_rterm1));                           \
  xmm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_rterm2, v_rterm3),                            \
                       _mm512_unpacklo_pd(v_rterm2, v_rterm3));                           \
  xmm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_rterm4, v_rterm5),                            \
                       _mm512_unpacklo_pd(v_rterm4, v_rterm5));                           \
  xmm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_rterm6, v_rterm7),                            \
                       _mm512_unpacklo_pd(v_rterm6, v_rterm7));                           \
                                                                                          \
  ymm0 = _mm512_add_pd(_mm512_permutex2var_pd(xmm0, permute_mask, xmm2),                  \
                       _mm512_mask_blend_pd(0xF0, xmm0, xmm2));                           \
                                                                                          \
  ymm1 = _mm512_add_pd(_mm512_permutex2var_pd(xmm1, permute_mask, xmm3),                  \
                       _mm512_mask_blend_pd(0xF0, xmm1, xmm3));                           \
                                                                                          \
  __m512d v_righterm_sum = _mm512_add_pd(_mm512_permutex2var_pd(ymm0,                     \
                                                                permute_mask_final_stage, \
                                                                ymm1),                    \
                                         _mm512_mask_blend_pd(0xCC, ymm0, ymm1));         \
                                                                                          \
  /* update sum */                                                                        \
  __m512d v_prod = _mm512_mul_pd(v_lefterm_sum, v_righterm_sum);                          \
                                                                                          \
  /* apply per-rate scalers */                                                            \
  if (rate_scalings && rate_scalings[i] > 0) {                                            \
    v_prod = _mm512_mul_pd(v_prod, v_scale_minlh[rate_scalings[i] - 1]);                  \
  }                                                                                       \
                                                                                          \
  _mm512_store_pd(sum + (j), v_prod);}                                                      \



#define COMPUTE_II_QROW_4(j) {                                                             \
/* point to the four rows of the eigenvecs matrix */                                       \
  const double *lm0 = ct_inv_eigenvecs;                                                    \
  const double *lm1 = lm0 + states_padded;                                                 \
  const double *lm2 = lm1 + states_padded;                                                 \
  const double *lm3 = lm2 + states_padded;                                                 \
  ct_inv_eigenvecs += ELEM_PER_AVX515_REGISTER * states_padded;                            \
                                                                                           \
  /* point to the four rows of the inv_eigenvecs matrix */                                 \
  const double *rm0 = c_eigenvecs;                                                         \
  const double *rm1 = rm0 + states_padded;                                                 \
  const double *rm2 = rm1 + states_padded;                                                 \
  const double *rm3 = rm2 + states_padded;                                                 \
  c_eigenvecs += ELEM_PER_AVX515_REGISTER * states_padded;                                 \
                                                                                           \
  __m512d v_lterm0 = _mm512_setzero_pd();                                                  \
  __m512d v_rterm0 = _mm512_setzero_pd();                                                  \
  __m512d v_lterm1 = _mm512_setzero_pd();                                                  \
  __m512d v_rterm1 = _mm512_setzero_pd();                                                  \
  __m512d v_lterm2 = _mm512_setzero_pd();                                                  \
  __m512d v_rterm2 = _mm512_setzero_pd();                                                  \
  __m512d v_lterm3 = _mm512_setzero_pd();                                                  \
  __m512d v_rterm3 = _mm512_setzero_pd();                                                  \
                                                                                           \
  __m512d v_mat;                                                                           \
                                                                                           \
  /* iterate over quadruples of columns */                                                 \
  COMPUTE_II_QCOL_HALF(0, 0);                                                              \
  COMPUTE_II_QCOL_HALF(1, 8);                                                              \
  COMPUTE_II_QCOL_HALF(2, 16);                                                             \
                                                                                           \
  /* compute lefterm */                                                                    \
  __m512d xmm0 = _mm512_unpackhi_pd(v_rterm0, v_rterm1);                                   \
  __m512d xmm1 = _mm512_unpacklo_pd(v_rterm0, v_rterm1);                                   \
                                                                                           \
  __m512d xmm2 = _mm512_unpackhi_pd(v_rterm2, v_rterm3);                                   \
  __m512d xmm3 = _mm512_unpacklo_pd(v_rterm2, v_rterm3);                                   \
                                                                                           \
  xmm0 = _mm512_add_pd(xmm0, xmm1);                                                        \
  xmm1 = _mm512_add_pd(xmm2, xmm3);                                                        \
                                                                                           \
  xmm2 = _mm512_permutex2var_pd(xmm0, permute_mask_final_stage, xmm1);                     \
                                                                                           \
  xmm3 = _mm512_mask_blend_pd(0xCC, xmm0, xmm1);                                           \
                                                                                           \
  __m512d blend = _mm512_add_pd(xmm2, xmm3);                                               \
                                                                                           \
  __m512d v_righterm_sum = _mm512_add_pd(blend,                                            \
                                      _mm512_permutexvar_pd(switch_256lanes_mask, blend)); \
                                                                                           \
  /* compute termb */                                                                      \
                                                                                           \
  xmm0 = _mm512_unpackhi_pd(v_lterm0, v_lterm1);                                           \
  xmm1 = _mm512_unpacklo_pd(v_lterm0, v_lterm1);                                           \
                                                                                           \
  xmm2 = _mm512_unpackhi_pd(v_lterm2, v_lterm3);                                           \
  xmm3 = _mm512_unpacklo_pd(v_lterm2, v_lterm3);                                           \
                                                                                           \
  xmm0 = _mm512_add_pd(xmm0, xmm1);                                                        \
  xmm1 = _mm512_add_pd(xmm2, xmm3);                                                        \
                                                                                           \
  xmm2 = _mm512_permutex2var_pd(xmm0, permute_mask_final_stage, xmm1);                     \
                                                                                           \
  xmm3 = _mm512_mask_blend_pd(0xCC, xmm0, xmm1);                                           \
                                                                                           \
  blend = _mm512_add_pd(xmm2, xmm3);                                                       \
                                                                                           \
  __m512d v_lefterm_sum = _mm512_add_pd(blend,                                             \
                                      _mm512_permutexvar_pd(switch_256lanes_mask, blend)); \
                                                                                           \
  __m512d v_prod = _mm512_mul_pd(v_righterm_sum, v_lefterm_sum);                           \
                                                                                           \
  /* apply per-rate scalers */                                                             \
  if (rate_scalings && rate_scalings[i] > 0) {                                             \
    v_prod = _mm512_mul_pd(v_prod, v_scale_minlh[rate_scalings[i] - 1]);                   \
  }                                                                                        \
                                                                                           \
  _mm512_store_pd(sum + (j), v_prod);}                                                       \


PLL_EXPORT int pll_core_update_sumtable_ii_20x20_avx512f(unsigned int sites,
                                                         unsigned int rate_cats,
                                                         const double *clvp,
                                                         const double *clvc,
                                                         const unsigned int *parent_scaler,
                                                         const unsigned int *child_scaler,
                                                         double *const *eigenvecs,
                                                         double *const *inv_eigenvecs,
                                                         double *const *freqs,
                                                         double *sumtable,
                                                         unsigned int attrib) {
  unsigned int i, j, k, n;

  const double *t_lclv = clvp;
  const double *t_rclv = clvc;
  double *t_freqs;

  unsigned int states = 20;
  unsigned int states_padded = (states + 7) & (0xFFFFFFFF - 7);

  /* build sumtable */
  //double *sum = sumtable;
  unsigned int sites_padded = (sites + ELEM_PER_AVX515_REGISTER - 1) & (0xFFFFFFFF - ELEM_PER_AVX515_REGISTER + 1);
  size_t sumtable_size = sites_padded * rate_cats * states_padded * sizeof(double);
  double *sumtable_copy = pll_aligned_alloc(sumtable_size, PLL_ALIGNMENT_AVX512F);
  memset(sumtable_copy, 0, sumtable_size);

  double *sum = sumtable_copy;

  __m512i switch_256lanes_mask = _mm512_setr_epi64(4,
                                                   5,
                                                   6,
                                                   7,
                                                   1,
                                                   2,
                                                   3,
                                                   4);
  __m512i permute_mask = _mm512_setr_epi64(0 | 4,
                                           0 | 5,
                                           0 | 6,
                                           0 | 7,
                                           8 | 0,
                                           8 | 1,
                                           8 | 2,
                                           8 | 3);
  __m512i permute_mask_final_stage = _mm512_setr_epi64(0 | 2,
                                                       0 | 3,
                                                       8 | 0,
                                                       8 | 1,
                                                       0 | 6,
                                                       0 | 7,
                                                       8 | 4,
                                                       8 | 5);

  /* scaling stuff */
  unsigned int min_scaler = 0;
  unsigned int *rate_scalings = NULL;
  int per_rate_scaling = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 1 : 0;

  /* powers of scale threshold for undoing the scaling */
  __m512d v_scale_minlh[PLL_SCALE_RATE_MAXDIFF];
  if (per_rate_scaling) {
    rate_scalings = (unsigned int *) calloc(rate_cats, sizeof(unsigned int));

    if (!rate_scalings) {
      pll_errno = PLL_ERROR_MEM_ALLOC;
      snprintf(pll_errmsg, 200, "Cannot allocate memory for rate scalers");
      return PLL_FAILURE;
    }

    double scale_factor = 1.0;
    for (i = 0; i < PLL_SCALE_RATE_MAXDIFF; ++i) {
      scale_factor *= PLL_SCALE_THRESHOLD;
      v_scale_minlh[i] = _mm512_set1_pd(scale_factor);
    }
  }

  /* padded eigenvecs */
  double *tt_eigenvecs = (double *) pll_aligned_alloc(
          (states_padded * states_padded * rate_cats) * sizeof(double),
          PLL_ALIGNMENT_AVX512F);

  if (!tt_eigenvecs) {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Cannot allocate memory for tt_eigenvecs");
    return PLL_FAILURE;
  }

  /* transposed padded inv_eigenvecs */
  double *tt_inv_eigenvecs = (double *) pll_aligned_alloc(
          (states_padded * states_padded * rate_cats) * sizeof(double),
          PLL_ALIGNMENT_AVX512F);

  if (!tt_inv_eigenvecs) {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Cannot allocate memory for tt_inv_eigenvecs");
    return PLL_FAILURE;
  }

  memset(tt_eigenvecs, 0, (states_padded * states_padded * rate_cats) * sizeof(double));
  memset(tt_inv_eigenvecs, 0, (states_padded * states_padded * rate_cats) * sizeof(double));

  /* add padding to eigenvecs matrices and multiply with frequencies */
  for (i = 0; i < rate_cats; ++i) {
    t_freqs = freqs[i];
    for (j = 0; j < states; ++j)
      for (k = 0; k < states; ++k) {
        tt_inv_eigenvecs[i * states_padded * states_padded + j * states_padded
                         + k] = inv_eigenvecs[i][k * states_padded + j] * t_freqs[k];
        tt_eigenvecs[i * states_padded * states_padded + j * states_padded
                     + k] = eigenvecs[i][j * states_padded + k];
      }
  }

  /* vectorized loop from update_sumtable() */
  for (n = 0; n < sites; n++) {
    /* compute per-rate scalers and obtain minimum value (within site) */
    if (per_rate_scaling) {
      min_scaler = UINT_MAX;
      for (i = 0; i < rate_cats; ++i) {
        rate_scalings[i] = (parent_scaler) ? parent_scaler[n * rate_cats + i] : 0;
        rate_scalings[i] += (child_scaler) ? child_scaler[n * rate_cats + i] : 0;
        if (rate_scalings[i] < min_scaler)
          min_scaler = rate_scalings[i];
      }

      /* compute relative capped per-rate scalers */
      for (i = 0; i < rate_cats; ++i) {
        rate_scalings[i] = PLL_MIN(rate_scalings[i] - min_scaler,
                                   PLL_SCALE_RATE_MAXDIFF);
      }
    }

    const double *c_eigenvecs = tt_eigenvecs;
    const double *ct_inv_eigenvecs = tt_inv_eigenvecs;
    for (i = 0; i < rate_cats; ++i) {
      __m512d v_lclv[3];
      __m512d v_rclv[3];
      for (j = 0; j < 3; ++j) {
        v_lclv[j] = _mm512_load_pd(t_lclv + j * ELEM_PER_AVX515_REGISTER);
        v_rclv[j] = _mm512_load_pd(t_rclv + j * ELEM_PER_AVX515_REGISTER);
      }

      COMPUTE_II_QROW_8(0);
      COMPUTE_II_QROW_8(8);
      COMPUTE_II_QROW_4(16);

      t_lclv += states_padded;
      t_rclv += states_padded;
      sum += states_padded;
    }
  }

  pll_aligned_free(tt_inv_eigenvecs);
  pll_aligned_free(tt_eigenvecs);
  if (rate_scalings)
    free(rate_scalings);

  memset(sumtable, 0, sites_padded * rate_cats * states * sizeof(double));

  unsigned int itr = 0;

  unsigned int elems_per_site = rate_cats * states_padded;
  unsigned int elems_per_site_octet = ELEM_PER_AVX515_REGISTER * elems_per_site;

  double *site0 = sumtable_copy;
  double *site1 = site0 + elems_per_site;
  double *site2 = site1 + elems_per_site;
  double *site3 = site2 + elems_per_site;
  double *site4 = site3 + elems_per_site;
  double *site5 = site4 + elems_per_site;
  double *site6 = site5 + elems_per_site;
  double *site7 = site6 + elems_per_site;

  unsigned int octet_offset = 0;
  for (n = 0; n < sites_padded / ELEM_PER_AVX515_REGISTER;
       n++, octet_offset += elems_per_site_octet) {
    unsigned int states_offset = 0;
    for (i = 0; i < rate_cats; i++, states_offset += states_padded) {
      for (j = 0; j < states; j++, itr += ELEM_PER_AVX515_REGISTER) {
        sumtable[itr + 0] = site0[j + octet_offset + states_offset];
        sumtable[itr + 1] = site1[j + octet_offset + states_offset];
        sumtable[itr + 2] = site2[j + octet_offset + states_offset];
        sumtable[itr + 3] = site3[j + octet_offset + states_offset];
        sumtable[itr + 4] = site4[j + octet_offset + states_offset];
        sumtable[itr + 5] = site5[j + octet_offset + states_offset];
        sumtable[itr + 6] = site6[j + octet_offset + states_offset];
        sumtable[itr + 7] = site7[j + octet_offset + states_offset];
      }
    }
  }

  pll_aligned_free(sumtable_copy);

  return PLL_SUCCESS;
}

PLL_EXPORT int pll_core_update_sumtable_ii_avx512f(unsigned int states,
                                                   unsigned int sites,
                                                   unsigned int rate_cats,
                                                   const double *clvp,
                                                   const double *clvc,
                                                   const unsigned int *parent_scaler,
                                                   const unsigned int *child_scaler,
                                                   double *const *eigenvecs,
                                                   double *const *inv_eigenvecs,
                                                   double *const *freqs,
                                                   double *sumtable,
                                                   unsigned int attrib) {
  unsigned int i, j, k, n;

  /* build sumtable */
  double *sum = sumtable;

  const double *t_clvp = clvp;
  const double *t_clvc = clvc;
  double *t_freqs;

  /* dedicated functions for 4x4 and 20x20 matrices */
  if (states == 4) {
    /* call AVX variant */
    return pll_core_update_sumtable_ii_avx(states,
                                           sites,
                                           rate_cats,
                                           clvp,
                                           clvc,
                                           parent_scaler,
                                           child_scaler,
                                           eigenvecs,
                                           inv_eigenvecs,
                                           freqs,
                                           sumtable,
                                           attrib);
  } else if (states == 20) {
    return pll_core_update_sumtable_ii_20x20_avx512f(sites,
                                                     rate_cats,
                                                     clvp,
                                                     clvc,
                                                     parent_scaler,
                                                     child_scaler,
                                                     eigenvecs,
                                                     inv_eigenvecs,
                                                     freqs,
                                                     sumtable,
                                                     attrib);
  }

  unsigned int states_padded = (states + 7) & (0xFFFFFFFF - 7);

  __m512i permute_mask = _mm512_setr_epi64(0 | 4,
                                           0 | 5,
                                           0 | 6,
                                           0 | 7,
                                           8 | 0,
                                           8 | 1,
                                           8 | 2,
                                           8 | 3);
  __m512i permute_mask_final_stage = _mm512_setr_epi64(0 | 2,
                                                       0 | 3,
                                                       8 | 0,
                                                       8 | 1,
                                                       0 | 6,
                                                       0 | 7,
                                                       8 | 4,
                                                       8 | 5);

  /* scaling stuff */
  unsigned int min_scaler = 0;
  unsigned int *rate_scalings = NULL;
  int per_rate_scaling = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 1 : 0;

  /* powers of scale threshold for undoing the scaling */
  __m512d v_scale_minlh[PLL_SCALE_RATE_MAXDIFF];
  if (per_rate_scaling) {
    rate_scalings = (unsigned int *) calloc(rate_cats, sizeof(unsigned int));

    if (!rate_scalings) {
      pll_errno = PLL_ERROR_MEM_ALLOC;
      snprintf(pll_errmsg, 200, "Cannot allocate memory for rate scalers");
      return PLL_FAILURE;
    }

    double scale_factor = 1.0;
    for (i = 0; i < PLL_SCALE_RATE_MAXDIFF; ++i) {
      scale_factor *= PLL_SCALE_THRESHOLD;
      v_scale_minlh[i] = _mm512_set1_pd(scale_factor);
    }
  }

  /* padded eigenvecs */
  double *tt_eigenvecs = (double *) pll_aligned_alloc(
          (states_padded * states_padded * rate_cats) * sizeof(double),
          PLL_ALIGNMENT_AVX);

  if (!tt_eigenvecs) {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Cannot allocate memory for tt_eigenvecs");
    return PLL_FAILURE;
  }

  /* transposed padded inv_eigenvecs */
  double *tt_inv_eigenvecs = (double *) pll_aligned_alloc(
          (states_padded * states_padded * rate_cats) * sizeof(double),
          PLL_ALIGNMENT_AVX);

  if (!tt_inv_eigenvecs) {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Cannot allocate memory for tt_inv_eigenvecs");
    return PLL_FAILURE;
  }

  memset(tt_eigenvecs, 0, (states_padded * states_padded * rate_cats) * sizeof(double));
  memset(tt_inv_eigenvecs, 0, (states_padded * states_padded * rate_cats) * sizeof(double));

  /* add padding to eigenvecs matrices and multiply with frequencies */
  for (i = 0; i < rate_cats; ++i) {
    t_freqs = freqs[i];
    for (j = 0; j < states; ++j)
      for (k = 0; k < states; ++k) {
        tt_inv_eigenvecs[i * states_padded * states_padded + j * states_padded
                         + k] = inv_eigenvecs[i][k * states_padded + j] * t_freqs[k];
        tt_eigenvecs[i * states_padded * states_padded + j * states_padded
                     + k] = eigenvecs[i][j * states_padded + k];
      }
  }

  /* vectorized loop from update_sumtable() */
  for (n = 0; n < sites; n++) {

    /* compute per-rate scalers and obtain minimum value (within site) */
    if (per_rate_scaling) {
      min_scaler = UINT_MAX;
      for (i = 0; i < rate_cats; ++i) {
        rate_scalings[i] = (parent_scaler) ? parent_scaler[n * rate_cats + i] : 0;
        rate_scalings[i] += (child_scaler) ? child_scaler[n * rate_cats + i] : 0;
        if (rate_scalings[i] < min_scaler)
          min_scaler = rate_scalings[i];
      }

      /* compute relative capped per-rate scalers */
      for (i = 0; i < rate_cats; ++i) {
        rate_scalings[i] = PLL_MIN(rate_scalings[i] - min_scaler,
                                   PLL_SCALE_RATE_MAXDIFF);
      }
    }

    const double *c_eigenvecs = tt_eigenvecs;
    const double *ct_inv_eigenvecs = tt_inv_eigenvecs;
    for (i = 0; i < rate_cats; ++i) {
      for (j = 0; j < states_padded; j += ELEM_PER_AVX515_REGISTER) {
        /* point to the eight rows of the eigenvecs matrix */
        const double *em0 = c_eigenvecs;
        const double *em1 = em0 + states_padded;
        const double *em2 = em1 + states_padded;
        const double *em3 = em2 + states_padded;
        const double *em4 = em3 + states_padded;
        const double *em5 = em4 + states_padded;
        const double *em6 = em5 + states_padded;
        const double *em7 = em6 + states_padded;
        c_eigenvecs += ELEM_PER_AVX515_REGISTER * states_padded;

        /* point to the eight rows of the inv_eigenvecs matrix */
        const double *im0 = ct_inv_eigenvecs;
        const double *im1 = im0 + states_padded;
        const double *im2 = im1 + states_padded;
        const double *im3 = im2 + states_padded;
        const double *im4 = im3 + states_padded;
        const double *im5 = im4 + states_padded;
        const double *im6 = im5 + states_padded;
        const double *im7 = im6 + states_padded;
        ct_inv_eigenvecs += ELEM_PER_AVX515_REGISTER * states_padded;

        __m512d v_lefterm0 = _mm512_setzero_pd();
        __m512d v_righterm0 = _mm512_setzero_pd();
        __m512d v_lefterm1 = _mm512_setzero_pd();
        __m512d v_righterm1 = _mm512_setzero_pd();
        __m512d v_lefterm2 = _mm512_setzero_pd();
        __m512d v_righterm2 = _mm512_setzero_pd();
        __m512d v_lefterm3 = _mm512_setzero_pd();
        __m512d v_righterm3 = _mm512_setzero_pd();
        __m512d v_lefterm4 = _mm512_setzero_pd();
        __m512d v_righterm4 = _mm512_setzero_pd();
        __m512d v_lefterm5 = _mm512_setzero_pd();
        __m512d v_righterm5 = _mm512_setzero_pd();
        __m512d v_lefterm6 = _mm512_setzero_pd();
        __m512d v_righterm6 = _mm512_setzero_pd();
        __m512d v_lefterm7 = _mm512_setzero_pd();
        __m512d v_righterm7 = _mm512_setzero_pd();

        __m512d v_eigen;
        __m512d v_clvp;
        __m512d v_clvc;

        for (k = 0; k < states_padded; k += ELEM_PER_AVX515_REGISTER) {
          v_clvp = _mm512_load_pd(t_clvp + k);
          v_clvc = _mm512_load_pd(t_clvc + k);

          /* row 0 */
          v_eigen = _mm512_load_pd(im0 + k);
          v_lefterm0 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm0);

          v_eigen = _mm512_load_pd(em0 + k);
          v_righterm0 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm0);

          /* row 1 */
          v_eigen = _mm512_load_pd(im1 + k);
          v_lefterm1 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm1);

          v_eigen = _mm512_load_pd(em1 + k);
          v_righterm1 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm1);

          /* row 2 */
          v_eigen = _mm512_load_pd(im2 + k);
          v_lefterm2 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm2);

          v_eigen = _mm512_load_pd(em2 + k);
          v_righterm2 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm2);

          /* row 3 */
          v_eigen = _mm512_load_pd(im3 + k);
          v_lefterm3 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm3);

          v_eigen = _mm512_load_pd(em3 + k);
          v_righterm3 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm3);

          /* row 4 */
          v_eigen = _mm512_load_pd(im4 + k);
          v_lefterm4 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm4);

          v_eigen = _mm512_load_pd(em4 + k);
          v_righterm4 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm4);

          /* row 5 */
          v_eigen = _mm512_load_pd(im5 + k);
          v_lefterm5 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm5);

          v_eigen = _mm512_load_pd(em5 + k);
          v_righterm5 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm5);

          /* row 6 */
          v_eigen = _mm512_load_pd(im6 + k);
          v_lefterm6 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm6);

          v_eigen = _mm512_load_pd(em6 + k);
          v_righterm6 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm6);

          /* row 7 */
          v_eigen = _mm512_load_pd(im7 + k);
          v_lefterm7 = _mm512_fmadd_pd(v_eigen, v_clvp, v_lefterm7);

          v_eigen = _mm512_load_pd(em7 + k);
          v_righterm7 = _mm512_fmadd_pd(v_eigen, v_clvc, v_righterm7);
        }

        /* compute lefterm */
        __m512d xmm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_lefterm0, v_lefterm1),
                                     _mm512_unpacklo_pd(v_lefterm0, v_lefterm1));
        __m512d xmm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_lefterm2, v_lefterm3),
                                     _mm512_unpacklo_pd(v_lefterm2, v_lefterm3));
        __m512d xmm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_lefterm4, v_lefterm5),
                                     _mm512_unpacklo_pd(v_lefterm4, v_lefterm5));
        __m512d xmm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_lefterm6, v_lefterm7),
                                     _mm512_unpacklo_pd(v_lefterm6, v_lefterm7));

        __m512d ymm0 = _mm512_add_pd(_mm512_permutex2var_pd(xmm0, permute_mask, xmm2),
                                     _mm512_mask_blend_pd(0xF0, xmm0, xmm2));

        __m512d ymm1 = _mm512_add_pd(_mm512_permutex2var_pd(xmm1, permute_mask, xmm3),
                                     _mm512_mask_blend_pd(0xF0, xmm1, xmm3));

        __m512d v_lefterm_sum = _mm512_add_pd(_mm512_permutex2var_pd(ymm0,
                                                                     permute_mask_final_stage,
                                                                     ymm1),
                                              _mm512_mask_blend_pd(0xCC, ymm0, ymm1));

        /* compute righterm */
        xmm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_righterm0, v_righterm1),
                             _mm512_unpacklo_pd(v_righterm0, v_righterm1));
        xmm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_righterm2, v_righterm3),
                             _mm512_unpacklo_pd(v_righterm2, v_righterm3));
        xmm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_righterm4, v_righterm5),
                             _mm512_unpacklo_pd(v_righterm4, v_righterm5));
        xmm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_righterm6, v_righterm7),
                             _mm512_unpacklo_pd(v_righterm6, v_righterm7));

        ymm0 = _mm512_add_pd(_mm512_permutex2var_pd(xmm0, permute_mask, xmm2),
                             _mm512_mask_blend_pd(0xF0, xmm0, xmm2));

        ymm1 = _mm512_add_pd(_mm512_permutex2var_pd(xmm1, permute_mask, xmm3),
                             _mm512_mask_blend_pd(0xF0, xmm1, xmm3));

        __m512d v_righterm_sum = _mm512_add_pd(_mm512_permutex2var_pd(ymm0,
                                                                      permute_mask_final_stage,
                                                                      ymm1),
                                               _mm512_mask_blend_pd(0xCC, ymm0, ymm1));

        /* update sum */
        __m512d v_prod = _mm512_mul_pd(v_lefterm_sum, v_righterm_sum);

        /* apply per-rate scalers */
        if (rate_scalings && rate_scalings[i] > 0) {
          v_prod = _mm512_mul_pd(v_prod, v_scale_minlh[rate_scalings[i] - 1]);
        }

        _mm512_store_pd(sum + j, v_prod);
      }

      t_clvc += states_padded;
      t_clvp += states_padded;
      sum += states_padded;
    }
  }

  pll_aligned_free(tt_inv_eigenvecs);
  pll_aligned_free(tt_eigenvecs);
  if (rate_scalings)
    free(rate_scalings);

  return PLL_SUCCESS;
}

PLL_EXPORT int pll_core_update_sumtable_ti_avx512f(unsigned int states,
                                                   unsigned int sites,
                                                   unsigned int rate_cats,
                                                   const double *parent_clv,
                                                   const unsigned char *left_tipchars,
                                                   const unsigned int *parent_scaler,
                                                   double *const *eigenvecs,
                                                   double *const *inv_eigenvecs,
                                                   double *const *freqs,
                                                   const unsigned int *tipmap,
                                                   unsigned int tipmap_size,
                                                   double *sumtable,
                                                   unsigned int attrib) {
  assert(0);
  //TODO: Not implemented!
  return PLL_FAILURE;
}

PLL_EXPORT
int pll_core_likelihood_derivatives_avx512f(unsigned int states,
                                            unsigned int states_padded,
                                            unsigned int rate_cats,
                                            unsigned int ef_sites,
                                            const unsigned int *pattern_weights,
                                            const double *rate_weights,
                                            const int *invariant,
                                            const double *prop_invar,
                                            double *const *freqs,
                                            const double *sumtable,
                                            const double *diagptable,
                                            double *d_f,
                                            double *dd_f) {
  /* vectors for accumulating LH, 1st and 2nd derivatives */
  __m512d v_df = _mm512_setzero_pd();
  __m512d v_ddf = _mm512_setzero_pd();
  __m512d v_all1 = _mm512_set1_pd(1.);

  __m512d site_lk[3];

  const double *sum = sumtable;
  const int *invariant_ptr = invariant;

  double *t_diagp = (double *) pll_aligned_alloc(
          ELEM_PER_AVX515_REGISTER * 3 * rate_cats * states * sizeof(double), PLL_ALIGNMENT_AVX512F);

  if (!t_diagp) {
    pll_errno = PLL_ERROR_MEM_ALLOC;
    snprintf(pll_errmsg, 200, "Unable to allocate enough memory.");
    return PLL_FAILURE;
  }

  /* transpose diagptable */
  for (unsigned int i = 0; i < rate_cats; ++i) {
    for (unsigned int j = 0; j < states; ++j) {
      for (unsigned int k = 0; k < 3; ++k) {
        __m512d v_diagp = _mm512_set1_pd(diagptable[i * states * 4 + j * 4 + k]);
        _mm512_store_pd(t_diagp +
                        i * ELEM_PER_AVX515_REGISTER * 3 * states +
                        j * ELEM_PER_AVX515_REGISTER * 3 +
                        k * ELEM_PER_AVX515_REGISTER,
                        v_diagp);
      }
    }
  }

  for (unsigned int n = 0;
       n < ef_sites;
       n += ELEM_PER_AVX515_REGISTER, invariant_ptr += ELEM_PER_AVX515_REGISTER) {
    site_lk[0] = _mm512_setzero_pd();
    site_lk[1] = _mm512_setzero_pd();
    site_lk[2] = _mm512_setzero_pd();

    const double *diagp = t_diagp;

    for (unsigned int i = 0; i < rate_cats; ++i) {

      __m512d v_cat_sitelk[3];
      v_cat_sitelk[0] = _mm512_setzero_pd();
      v_cat_sitelk[1] = _mm512_setzero_pd();
      v_cat_sitelk[2] = _mm512_setzero_pd();

      for (unsigned int j = 0;
           j < states; j++, diagp += 3 * ELEM_PER_AVX515_REGISTER, sum += ELEM_PER_AVX515_REGISTER) {
        __m512d v_sum = _mm512_load_pd(sum);
        __m512d v_diagp;

        v_diagp = _mm512_load_pd(diagp);
        //v_diagp = _mm512_set1_pd(diagp[0]);
        v_cat_sitelk[0] = _mm512_fmadd_pd(v_sum, v_diagp, v_cat_sitelk[0]);

        v_diagp = _mm512_load_pd(diagp + ELEM_PER_AVX515_REGISTER);
        v_cat_sitelk[1] = _mm512_fmadd_pd(v_sum, v_diagp, v_cat_sitelk[1]);

        v_diagp = _mm512_load_pd(diagp + 2 * ELEM_PER_AVX515_REGISTER);
        v_cat_sitelk[2] = _mm512_fmadd_pd(v_sum, v_diagp, v_cat_sitelk[2]);
      }

      /* account for invariant sites */
      double t_prop_invar = prop_invar[i];
      if (t_prop_invar > 0) {

        //TODO Vectorize?
        double inv_site_lk_0 =
                (n + 0 >= ef_sites || invariant_ptr[0] == -1) ? 0 : freqs[i][invariant_ptr[0]] * t_prop_invar;
        double inv_site_lk_1 =
                (n + 1 >= ef_sites || invariant_ptr[1] == -1) ? 0 : freqs[i][invariant_ptr[1]] * t_prop_invar;
        double inv_site_lk_2 =
                (n + 2 >= ef_sites || invariant_ptr[2] == -1) ? 0 : freqs[i][invariant_ptr[2]] * t_prop_invar;
        double inv_site_lk_3 =
                (n + 3 >= ef_sites || invariant_ptr[3] == -1) ? 0 : freqs[i][invariant_ptr[3]] * t_prop_invar;
        double inv_site_lk_4 =
                (n + 4 >= ef_sites || invariant_ptr[4] == -1) ? 0 : freqs[i][invariant_ptr[4]] * t_prop_invar;
        double inv_site_lk_5 =
                (n + 5 >= ef_sites || invariant_ptr[5] == -1) ? 0 : freqs[i][invariant_ptr[5]] * t_prop_invar;
        double inv_site_lk_6 =
                (n + 6 >= ef_sites || invariant_ptr[6] == -1) ? 0 : freqs[i][invariant_ptr[6]] * t_prop_invar;
        double inv_site_lk_7 =
                (n + 7 >= ef_sites || invariant_ptr[7] == -1) ? 0 : freqs[i][invariant_ptr[7]] * t_prop_invar;

        __m512d v_inv_site_lk = _mm512_setr_pd(inv_site_lk_0,
                                               inv_site_lk_1,
                                               inv_site_lk_2,
                                               inv_site_lk_3,
                                               inv_site_lk_4,
                                               inv_site_lk_5,
                                               inv_site_lk_6,
                                               inv_site_lk_7);

        __m512d v_prop_invar = _mm512_set1_pd(1. - t_prop_invar);

        v_cat_sitelk[0] = _mm512_add_pd(_mm512_mul_pd(v_cat_sitelk[0], v_prop_invar), v_inv_site_lk);
        v_cat_sitelk[1] = _mm512_mul_pd(v_cat_sitelk[1], v_prop_invar);
        v_cat_sitelk[2] = _mm512_mul_pd(v_cat_sitelk[2], v_prop_invar);
      }

      /* apply rate category weights */
      __m512d v_weight = _mm512_set1_pd(rate_weights[i]);
      site_lk[0] = _mm512_fmadd_pd(v_cat_sitelk[0], v_weight, site_lk[0]);
      site_lk[1] = _mm512_fmadd_pd(v_cat_sitelk[1], v_weight, site_lk[1]);
      site_lk[2] = _mm512_fmadd_pd(v_cat_sitelk[2], v_weight, site_lk[2]);
    }

    /* build derivatives */
    __m512d v_recip0 = _mm512_div_pd(v_all1, site_lk[0]);
    __m512d v_deriv1 = _mm512_mul_pd(site_lk[1], v_recip0);
    __m512d v_deriv2 = _mm512_sub_pd(_mm512_mul_pd(v_deriv1, v_deriv1),
                                     _mm512_mul_pd(site_lk[2], v_recip0));

    /* eliminates nan values on padded states */
    if (n + ELEM_PER_AVX515_REGISTER > ef_sites) {
      __mmask8 mask = _mm512_cmp_pd_mask(site_lk[0], _mm512_setzero_pd(), _CMP_NEQ_UQ);

      v_deriv1 = _mm512_maskz_expand_pd(mask, v_deriv1);
      v_deriv2 = _mm512_maskz_expand_pd(mask, v_deriv2);
    }

    v_df = _mm512_fnmadd_pd(v_deriv1, _mm512_set1_pd(pattern_weights[n]), v_df);
    v_ddf = _mm512_fmadd_pd(v_deriv2, _mm512_set1_pd(pattern_weights[n]), v_ddf);
  }

  *d_f = reduce_add_pd(v_df);
  *dd_f = reduce_add_pd(v_ddf);

  return PLL_SUCCESS;
}