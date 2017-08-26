/*
    Copyright (C) 2016 Tomas Flouri, Kassian Kobert

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

#include "pll.h"

#define ELEM_PER_AVX515_REGISTER 8


#define PROCESS_8_COLS_HALF(j)                         \
  v_lclv = _mm512_load_pd(left_clv + j);               \
  v_rclv = _mm512_load_pd(right_clv + j);              \
                                                       \
  /* row 0 */                                          \
  v_mat    = _mm512_load_pd(lm0);                      \
  v_terma0 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma0); \
  v_mat    = _mm512_load_pd(rm0);                      \
  v_termb0 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb0); \
  lm0 += ELEM_PER_AVX515_REGISTER;                     \
  rm0 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 1 */                                          \
  v_mat    = _mm512_load_pd(lm1);                      \
  v_terma1 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma1); \
  v_mat    = _mm512_load_pd(rm1);                      \
  v_termb1 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb1); \
  lm1 += ELEM_PER_AVX515_REGISTER;                     \
  rm1 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 2 */                                          \
  v_mat    = _mm512_load_pd(lm2);                      \
  v_terma2 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma2); \
  v_mat    = _mm512_load_pd(rm2);                      \
  v_termb2 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb2); \
  lm2 += ELEM_PER_AVX515_REGISTER;                     \
  rm2 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 3 */                                          \
  v_mat    = _mm512_load_pd(lm3);                      \
  v_terma3 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma3); \
  v_mat    = _mm512_load_pd(rm3);                      \
  v_termb3 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb3); \
  lm3 += ELEM_PER_AVX515_REGISTER;                     \
  rm3 += ELEM_PER_AVX515_REGISTER;                     \

#define PROCESS_8_COLS_FULL(j)                         \
  v_lclv = _mm512_load_pd(left_clv + j);               \
  v_rclv = _mm512_load_pd(right_clv + j);              \
                                                       \
  /* row 0 */                                          \
  v_mat    = _mm512_load_pd(lm0);                      \
  v_terma0 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma0); \
  v_mat    = _mm512_load_pd(rm0);                      \
  v_termb0 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb0); \
  lm0  += ELEM_PER_AVX515_REGISTER;                    \
  rm0  += ELEM_PER_AVX515_REGISTER;                    \
                                                       \
  /* row 1 */                                          \
  v_mat    = _mm512_load_pd(lm1);                      \
  v_terma1 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma1); \
  v_mat    = _mm512_load_pd(rm1);                      \
  v_termb1 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb1); \
  lm1 += ELEM_PER_AVX515_REGISTER;                     \
  rm1 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 2 */                                          \
  v_mat    = _mm512_load_pd(lm2);                      \
  v_terma2 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma2); \
  v_mat    = _mm512_load_pd(rm2);                      \
  v_termb2 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb2); \
  lm2 += ELEM_PER_AVX515_REGISTER;                     \
  rm2 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 3 */                                          \
  v_mat    = _mm512_load_pd(lm3);                      \
  v_terma3 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma3); \
  v_mat    = _mm512_load_pd(rm3);                      \
  v_termb3 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb3); \
  lm3 += ELEM_PER_AVX515_REGISTER;                     \
  rm3 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 4 */                                          \
  v_mat    = _mm512_load_pd(lm4);                      \
  v_terma4 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma4); \
  v_mat    = _mm512_load_pd(rm4);                      \
  v_termb4 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb4); \
  lm4 += ELEM_PER_AVX515_REGISTER;                     \
  rm4 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 5 */                                          \
  v_mat    = _mm512_load_pd(lm5);                      \
  v_terma5 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma5); \
  v_mat    = _mm512_load_pd(rm5);                      \
  v_termb5 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb5); \
  lm5 += ELEM_PER_AVX515_REGISTER;                     \
  rm5 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 6 */                                          \
  v_mat    = _mm512_load_pd(lm6);                      \
  v_terma6 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma6); \
  v_mat    = _mm512_load_pd(rm6);                      \
  v_termb6 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb6); \
  lm6 += ELEM_PER_AVX515_REGISTER;                     \
  rm6 += ELEM_PER_AVX515_REGISTER;                     \
                                                       \
  /* row 7 */                                          \
  v_mat    = _mm512_load_pd(lm7);                      \
  v_terma7 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma7); \
  v_mat    = _mm512_load_pd(rm7);                      \
  v_termb7 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb7); \
  lm7 += ELEM_PER_AVX515_REGISTER;                     \
  rm7 += ELEM_PER_AVX515_REGISTER;                     \


#define PROCESS_8_ROWS(i) {                                                            \
  __m512d v_terma0 = _mm512_setzero_pd();                                              \
  __m512d v_termb0 = _mm512_setzero_pd();                                              \
  __m512d v_terma1 = _mm512_setzero_pd();                                              \
  __m512d v_termb1 = _mm512_setzero_pd();                                              \
  __m512d v_terma2 = _mm512_setzero_pd();                                              \
  __m512d v_termb2 = _mm512_setzero_pd();                                              \
  __m512d v_terma3 = _mm512_setzero_pd();                                              \
  __m512d v_termb3 = _mm512_setzero_pd();                                              \
  __m512d v_terma4 = _mm512_setzero_pd();                                              \
  __m512d v_termb4 = _mm512_setzero_pd();                                              \
  __m512d v_terma5 = _mm512_setzero_pd();                                              \
  __m512d v_termb5 = _mm512_setzero_pd();                                              \
  __m512d v_terma6 = _mm512_setzero_pd();                                              \
  __m512d v_termb6 = _mm512_setzero_pd();                                              \
  __m512d v_terma7 = _mm512_setzero_pd();                                              \
  __m512d v_termb7 = _mm512_setzero_pd();                                              \
                                                                                       \
  __m512d v_mat;                                                                       \
  __m512d v_lclv;                                                                      \
  __m512d v_rclv;                                                                      \
                                                                                       \
  /* point to the eight rows of the left matrix */                                     \
  const double *lm0 = lmat;                                                            \
  const double *lm1 = lm0 + states_padded;                                             \
  const double *lm2 = lm1 + states_padded;                                             \
  const double *lm3 = lm2 + states_padded;                                             \
  const double *lm4 = lm3 + states_padded;                                             \
  const double *lm5 = lm4 + states_padded;                                             \
  const double *lm6 = lm5 + states_padded;                                             \
  const double *lm7 = lm6 + states_padded;                                             \
                                                                                       \
  /* point to the eight rows of the right matrix */                                    \
  const double *rm0 = rmat;                                                            \
  const double *rm1 = rm0 + states_padded;                                             \
  const double *rm2 = rm1 + states_padded;                                             \
  const double *rm3 = rm2 + states_padded;                                             \
  const double *rm4 = rm3 + states_padded;                                             \
  const double *rm5 = rm4 + states_padded;                                             \
  const double *rm6 = rm5 + states_padded;                                             \
  const double *rm7 = rm6 + states_padded;                                             \
                                                                                       \
  PROCESS_8_COLS_FULL(0);                                                              \
  PROCESS_8_COLS_FULL(8);                                                              \
  PROCESS_8_COLS_FULL(16);                                                             \
                                                                                       \
  /* point pmatrix to the next four rows */                                            \
  lmat = lm7;                                                                          \
  rmat = rm7;                                                                          \
                                                                                       \
  __m512d ymm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma0, v_terma1),                 \
                               _mm512_unpacklo_pd(v_terma0, v_terma1));                \
  __m512d ymm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma2, v_terma3),                 \
                               _mm512_unpacklo_pd(v_terma2, v_terma3));                \
  __m512d ymm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma4, v_terma5),                 \
                               _mm512_unpacklo_pd(v_terma4, v_terma5));                \
  __m512d ymm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma6, v_terma7),                 \
                               _mm512_unpacklo_pd(v_terma6, v_terma7));                \
                                                                                       \
  __m512d zmm0 = _mm512_add_pd(_mm512_permutex2var_pd(ymm0, permute_mask, ymm2),       \
                               _mm512_mask_blend_pd(0xF0, ymm0, ymm2));                \
                                                                                       \
  __m512d zmm1 = _mm512_add_pd(_mm512_permutex2var_pd(ymm1, permute_mask, ymm3),       \
                               _mm512_mask_blend_pd(0xF0, ymm1, ymm3));                \
                                                                                       \
  __m512d v_terma_sum = _mm512_add_pd(_mm512_permutex2var_pd(zmm0,                     \
                                                             permute_mask_final_stage, \
                                                             zmm1),                    \
                                      _mm512_mask_blend_pd(0xCC, zmm0, zmm1));         \
                                                                                       \
  /* compute termb */                                                                  \
                                                                                       \
  ymm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb0, v_termb1),                         \
                       _mm512_unpacklo_pd(v_termb0, v_termb1));                        \
  ymm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb2, v_termb3),                         \
                       _mm512_unpacklo_pd(v_termb2, v_termb3));                        \
  ymm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb4, v_termb5),                         \
                       _mm512_unpacklo_pd(v_termb4, v_termb5));                        \
  ymm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb6, v_termb7),                         \
                       _mm512_unpacklo_pd(v_termb6, v_termb7));                        \
                                                                                       \
  zmm0 = _mm512_add_pd(_mm512_permutex2var_pd(ymm0, permute_mask, ymm2),               \
                       _mm512_mask_blend_pd(0xF0, ymm0, ymm2));                        \
                                                                                       \
  zmm1 = _mm512_add_pd(_mm512_permutex2var_pd(ymm1, permute_mask, ymm3),               \
                       _mm512_mask_blend_pd(0xF0, ymm1, ymm3));                        \
                                                                                       \
  __m512d v_termb_sum = _mm512_add_pd(_mm512_permutex2var_pd(zmm0,                     \
                                                             permute_mask_final_stage, \
                                                             zmm1),                    \
                                      _mm512_mask_blend_pd(0xCC, zmm0, zmm1));         \
                                                                                       \
  __m512d v_prod = _mm512_mul_pd(v_terma_sum, v_termb_sum);                            \
                                                                                       \
  /* check if scaling is needed for the current rate category */                       \
  rate_mask = _mm512_cmp_pd_mask(v_prod, v_scale_threshold, _CMP_LT_OS);               \
                                                                                       \
  _mm512_store_pd(parent_clv + i, v_prod);}                                            \


void print_512d(const __m512d v) {
  const double *val = (const double *) &v;
  printf("% f % f % f % f % f % f % f % f\n",
         val[7], val[6], val[5], val[4], val[3], val[2], val[1], val[0]);
}

static void fill_parent_scaler(unsigned int scaler_size,
                               unsigned int *parent_scaler,
                               const unsigned int *left_scaler,
                               const unsigned int *right_scaler) {
  unsigned int i;

  if (!left_scaler && !right_scaler)
    memset(parent_scaler, 0, sizeof(unsigned int) * scaler_size);
  else if (left_scaler && right_scaler) {
    memcpy(parent_scaler, left_scaler, sizeof(unsigned int) * scaler_size);
    for (i = 0; i < scaler_size; ++i)
      parent_scaler[i] += right_scaler[i];
  } else {
    if (left_scaler)
      memcpy(parent_scaler, left_scaler, sizeof(unsigned int) * scaler_size);
    else
      memcpy(parent_scaler, right_scaler, sizeof(unsigned int) * scaler_size);
  }
}


PLL_EXPORT void pll_core_update_partial_ti_avx512f(unsigned int states,
                                                   unsigned int sites,
                                                   unsigned int rate_cats,
                                                   double *parent_clv,
                                                   unsigned int *parent_scaler,
                                                   const unsigned char *left_tipchars,
                                                   const double *right_clv,
                                                   const double *left_matrix,
                                                   const double *right_matrix,
                                                   const unsigned int *right_scaler,
                                                   const unsigned int *tipmap,
                                                   unsigned int tipmap_size,
                                                   unsigned int attrib) {
  assert(0);
//  unsigned int i, j, k, n;
//
//  const double *lmat;
//  const double *rmat;
//
//  unsigned int states_padded = (states + 3) & 0xFFFFFFFC;
//  unsigned int span_padded = states_padded * rate_cats;
//
//  unsigned int lstate;
//
//  /* dedicated functions for 4x4 matrices (DNA) */
//  if (states == 4) {
//    /* no AVX2 kernel so far; rollback to AVX */
//    pll_core_update_partial_ti_4x4_avx(sites,
//                                       rate_cats,
//                                       parent_clv,
//                                       parent_scaler,
//                                       left_tipchars,
//                                       right_clv,
//                                       left_matrix,
//                                       right_matrix,
//                                       right_scaler,
//                                       attrib);
//    return;
//  }
//
//  /* dedicated functions for 20x20 matrices (AA) */
//  if (states == 20) {
//    pll_core_update_partial_ti_20x20_avx512f(sites,
//                                             rate_cats,
//                                             parent_clv,
//                                             parent_scaler,
//                                             left_tipchars,
//                                             right_clv,
//                                             left_matrix,
//                                             right_matrix,
//                                             right_scaler,
//                                             tipmap,
//                                             tipmap_size,
//                                             attrib);
//    return;
//  }
//
//  /* scaling-related stuff */
//  unsigned int scale_mode;  /* 0 = none, 1 = per-site, 2 = per-rate */
//  unsigned int scale_mask;
//  unsigned int init_mask;
//  __m256d v_scale_threshold = _mm256_set1_pd(PLL_SCALE_THRESHOLD);
//  __m256d v_scale_factor = _mm256_set1_pd(PLL_SCALE_FACTOR);
//
//  if (!parent_scaler) {
//    /* scaling disabled / not required */
//    scale_mode = init_mask = 0;
//  } else {
//    /* determine the scaling mode and init the vars accordingly */
//    scale_mode = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 2 : 1;
//    init_mask = (scale_mode == 1) ? 0xF : 0;
//    const size_t scaler_size = (scale_mode == 2) ? sites * rate_cats : sites;
//    /* add up the scale vector of the two children if available */
//    fill_parent_scaler(scaler_size, parent_scaler, NULL, right_scaler);
//  }
//
//  size_t displacement = (states_padded - states) * (states_padded);
//
//  __m256i mask;
//
//  /* compute CLV */
//  for (n = 0; n < sites; ++n) {
//    lmat = left_matrix;
//    rmat = right_matrix;
//
//    scale_mask = init_mask;
//
//    lstate = tipmap[left_tipchars[n]];
//
//    for (k = 0; k < rate_cats; ++k) {
//      unsigned int rate_mask = 0xF;
//
//      /* iterate over quadruples of rows */
//      for (i = 0; i < states_padded; i += 4) {
//
//        __m256d v_terma0 = _mm256_setzero_pd();
//        __m256d v_termb0 = _mm256_setzero_pd();
//        __m256d v_terma1 = _mm256_setzero_pd();
//        __m256d v_termb1 = _mm256_setzero_pd();
//        __m256d v_terma2 = _mm256_setzero_pd();
//        __m256d v_termb2 = _mm256_setzero_pd();
//        __m256d v_terma3 = _mm256_setzero_pd();
//        __m256d v_termb3 = _mm256_setzero_pd();
//
//        __m256d v_mat;
//        __m256d v_rclv;
//
//        /* point to the four rows of the left matrix */
//        const double *lm0 = lmat;
//        const double *lm1 = lm0 + states_padded;
//        const double *lm2 = lm1 + states_padded;
//        const double *lm3 = lm2 + states_padded;
//
//        /* point to the four rows of the right matrix */
//        const double *rm0 = rmat;
//        const double *rm1 = rm0 + states_padded;
//        const double *rm2 = rm1 + states_padded;
//        const double *rm3 = rm2 + states_padded;
//
//        /* set position of least significant bit in character state */
//        register int lsb = 0;
//
//        /* iterate over quadruples of columns */
//        for (j = 0; j < states_padded; j += 4) {
//
//          /* set mask */
//          mask = _mm256_set_epi64x(
//                  ((lstate >> (lsb + 3)) & 1) ? ~0 : 0,
//                  ((lstate >> (lsb + 2)) & 1) ? ~0 : 0,
//                  ((lstate >> (lsb + 1)) & 1) ? ~0 : 0,
//                  ((lstate >> (lsb + 0)) & 1) ? ~0 : 0);
//
//          if ((lstate >> lsb) & 0b1111) {
//            v_mat = _mm256_maskload_pd(lm0, mask);
//            v_terma0 = _mm256_add_pd(v_terma0, v_mat);
//
//            v_mat = _mm256_maskload_pd(lm1, mask);
//            v_terma1 = _mm256_add_pd(v_terma1, v_mat);
//
//            v_mat = _mm256_maskload_pd(lm2, mask);
//            v_terma2 = _mm256_add_pd(v_terma2, v_mat);
//
//            v_mat = _mm256_maskload_pd(lm3, mask);
//            v_terma3 = _mm256_add_pd(v_terma3, v_mat);
//          }
//
//          lsb += 4;
//
//          lm0 += 4;
//          lm1 += 4;
//          lm2 += 4;
//          lm3 += 4;
//
//          v_rclv = _mm256_load_pd(right_clv + j);
//
//          /* row 0 */
//          v_mat = _mm256_load_pd(rm0);
//          v_termb0 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb0);
//          rm0 += 4;
//
//          /* row 1 */
//          v_mat = _mm256_load_pd(rm1);
//          v_termb1 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb1);
//          rm1 += 4;
//
//          /* row 2 */
//          v_mat = _mm256_load_pd(rm2);
//          v_termb2 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb2);
//          rm2 += 4;
//
//          /* row 3 */
//          v_mat = _mm256_load_pd(rm3);
//          v_termb3 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb3);
//          rm3 += 4;
//        }
//
//        /* point pmatrix to the next four rows */
//        lmat = lm3;
//        rmat = rm3;
//
//        __m256d xmm0 = _mm256_unpackhi_pd(v_terma0, v_terma1);
//        __m256d xmm1 = _mm256_unpacklo_pd(v_terma0, v_terma1);
//
//        __m256d xmm2 = _mm256_unpackhi_pd(v_terma2, v_terma3);
//        __m256d xmm3 = _mm256_unpacklo_pd(v_terma2, v_terma3);
//
//        xmm0 = _mm256_add_pd(xmm0, xmm1);
//        xmm1 = _mm256_add_pd(xmm2, xmm3);
//
//        xmm2 = _mm256_permute2f128_pd(xmm0, xmm1, _MM_SHUFFLE(0, 2, 0, 1));
//
//        xmm3 = _mm256_blend_pd(xmm0, xmm1, 12);
//
//        __m256d v_terma_sum = _mm256_add_pd(xmm2, xmm3);
//
//        /* compute termb */
//
//        xmm0 = _mm256_unpackhi_pd(v_termb0, v_termb1);
//        xmm1 = _mm256_unpacklo_pd(v_termb0, v_termb1);
//
//        xmm2 = _mm256_unpackhi_pd(v_termb2, v_termb3);
//        xmm3 = _mm256_unpacklo_pd(v_termb2, v_termb3);
//
//        xmm0 = _mm256_add_pd(xmm0, xmm1);
//        xmm1 = _mm256_add_pd(xmm2, xmm3);
//
//        xmm2 = _mm256_permute2f128_pd(xmm0, xmm1, _MM_SHUFFLE(0, 2, 0, 1));
//
//        xmm3 = _mm256_blend_pd(xmm0, xmm1, 12);
//
//        __m256d v_termb_sum = _mm256_add_pd(xmm2, xmm3);
//
//        __m256d v_prod = _mm256_mul_pd(v_terma_sum, v_termb_sum);
//
//        /* check if scaling is needed for the current rate category */
//        __m256d v_cmp = _mm256_cmp_pd(v_prod, v_scale_threshold, _CMP_LT_OS);
//        rate_mask = rate_mask & _mm256_movemask_pd(v_cmp);
//
//        _mm256_store_pd(parent_clv + i, v_prod);
//
//      }
//
//      if (scale_mode == 2) {
//        /* PER-RATE SCALING: if *all* entries of the *rate* CLV were below
//         * the threshold then scale (all) entries by PLL_SCALE_FACTOR */
//        if (rate_mask == 0xF) {
//          for (i = 0; i < states_padded; i += 4) {
//            __m256d v_prod = _mm256_load_pd(parent_clv + i);
//            v_prod = _mm256_mul_pd(v_prod, v_scale_factor);
//            _mm256_store_pd(parent_clv + i, v_prod);
//          }
//          parent_scaler[n * rate_cats + k] += 1;
//        }
//      } else
//        scale_mask = scale_mask & rate_mask;
//
//      /* reset pointers to point to the start of the next p-matrix, as the
//         vectorization assumes a square states_padded * states_padded matrix,
//         even though the real matrix is states * states_padded */
//      lmat -= displacement;
//      rmat -= displacement;
//
//      parent_clv += states_padded;
//      right_clv += states_padded;
//    }
//
//    /* if *all* entries of the site CLV were below the threshold then scale
//       (all) entries by PLL_SCALE_FACTOR */
//    if (scale_mask == 0xF) {
//      parent_clv -= span_padded;
//      for (i = 0; i < span_padded; i += 4) {
//        __m256d v_prod = _mm256_load_pd(parent_clv + i);
//        v_prod = _mm256_mul_pd(v_prod, v_scale_factor);
//        _mm256_store_pd(parent_clv + i, v_prod);
//      }
//      parent_clv += span_padded;
//      parent_scaler[n] += 1;
//    }
//  }
}


PLL_EXPORT
void pll_core_update_partial_ii_20x20_avx512f(unsigned int sites,
                                              unsigned int rate_cats,
                                              double *parent_clv,
                                              unsigned int *parent_scaler,
                                              const double *left_clv,
                                              const double *right_clv,
                                              const double *left_matrix,
                                              const double *right_matrix,
                                              const unsigned int *left_scaler,
                                              const unsigned int *right_scaler,
                                              unsigned int attrib) {
  unsigned int i, k, n;

  const double *lmat;
  const double *rmat;

  unsigned int states = 20;
  unsigned int states_padded = (states + 7) & (0xFFFFFFFF - 7);
  unsigned int span_padded = states_padded * rate_cats;

  /* scaling-related stuff */
  unsigned int scale_mode;  /* 0 = none, 1 = per-site, 2 = per-rate */
  unsigned int scale_mask;
  unsigned int init_mask;
  __m512d v_scale_threshold = _mm512_set1_pd(PLL_SCALE_THRESHOLD);
  __m512d v_scale_factor = _mm512_set1_pd(PLL_SCALE_FACTOR);

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

  if (!parent_scaler) {
    /* scaling disabled / not required */
    scale_mode = init_mask = 0;
  } else {
    /* determine the scaling mode and init the vars accordingly */
    scale_mode = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 2 : 1;
    init_mask = (scale_mode == 1) ? 0xF : 0;
    const size_t scaler_size = (scale_mode == 2) ? sites * rate_cats : sites;
    /* add up the scale vector of the two children if available */
    fill_parent_scaler(scaler_size, parent_scaler, left_scaler, right_scaler);
  }

  size_t displacement = (states_padded - states) * (states_padded);

  /* compute CLV */
  for (n = 0; n < sites; ++n) {
    lmat = left_matrix;
    rmat = right_matrix;
    scale_mask = init_mask;

    for (k = 0; k < rate_cats; ++k) {
      __mmask8 rate_mask = 0xF;

      PROCESS_8_ROWS(0);
      PROCESS_8_ROWS(8);

      __m512d v_terma0 = _mm512_setzero_pd();
      __m512d v_termb0 = _mm512_setzero_pd();
      __m512d v_terma1 = _mm512_setzero_pd();
      __m512d v_termb1 = _mm512_setzero_pd();
      __m512d v_terma2 = _mm512_setzero_pd();
      __m512d v_termb2 = _mm512_setzero_pd();
      __m512d v_terma3 = _mm512_setzero_pd();
      __m512d v_termb3 = _mm512_setzero_pd();

      __m512d v_mat;
      __m512d v_lclv;
      __m512d v_rclv;

      /* point to the four rows of the left matrix */
      const double *lm0 = lmat;
      const double *lm1 = lm0 + states_padded;
      const double *lm2 = lm1 + states_padded;
      const double *lm3 = lm2 + states_padded;

      /* point to the four rows of the right matrix */
      const double *rm0 = rmat;
      const double *rm1 = rm0 + states_padded;
      const double *rm2 = rm1 + states_padded;
      const double *rm3 = rm2 + states_padded;

      PROCESS_8_COLS_HALF(0);
      PROCESS_8_COLS_HALF(8);
      PROCESS_8_COLS_HALF(16);

      /* point pmatrix to the next four rows */
      lmat += 8 * states_padded;
      rmat += 8 * states_padded;

      __m512d xmm0 = _mm512_unpackhi_pd(v_terma0, v_terma1);
      __m512d xmm1 = _mm512_unpacklo_pd(v_terma0, v_terma1);

      __m512d xmm2 = _mm512_unpackhi_pd(v_terma2, v_terma3);
      __m512d xmm3 = _mm512_unpacklo_pd(v_terma2, v_terma3);

      xmm0 = _mm512_add_pd(xmm0, xmm1);
      xmm1 = _mm512_add_pd(xmm2, xmm3);

      xmm2 = _mm512_permutex2var_pd(xmm0, permute_mask_final_stage, xmm1);

      xmm3 = _mm512_mask_blend_pd(0xCC, xmm0, xmm1);

      __m512d blend = _mm512_add_pd(xmm2, xmm3);

      __m512d v_terma_sum = _mm512_add_pd(blend, _mm512_permutexvar_pd(switch_256lanes_mask, blend));

      /* compute termb */

      xmm0 = _mm512_unpackhi_pd(v_termb0, v_termb1);
      xmm1 = _mm512_unpacklo_pd(v_termb0, v_termb1);

      xmm2 = _mm512_unpackhi_pd(v_termb2, v_termb3);
      xmm3 = _mm512_unpacklo_pd(v_termb2, v_termb3);

      xmm0 = _mm512_add_pd(xmm0, xmm1);
      xmm1 = _mm512_add_pd(xmm2, xmm3);

      xmm2 = _mm512_permutex2var_pd(xmm0, permute_mask_final_stage, xmm1);

      xmm3 = _mm512_mask_blend_pd(0xCC, xmm0, xmm1);

      blend = _mm512_add_pd(xmm2, xmm3);

      __m512d v_termb_sum = _mm512_add_pd(blend, _mm512_permutexvar_pd(switch_256lanes_mask, blend));

      __m512d v_prod = _mm512_mul_pd(v_terma_sum, v_termb_sum);

      /* check if scaling is needed for the current rate category */
      rate_mask = _mm512_cmp_pd_mask(v_prod, v_scale_threshold, _CMP_LT_OS);

      _mm512_store_pd(parent_clv + 16, v_prod);

      if (scale_mode == 2) {
        /* PER-RATE SCALING: if *all* entries of the *rate* CLV were below
       * the threshold then scale (all) entries by PLL_SCALE_FACTOR */
        if (rate_mask == 0xF) {
          for (i = 0; i < states_padded; i += ELEM_PER_AVX515_REGISTER) {
            __m512d v_prod = _mm512_load_pd(parent_clv + i);
            v_prod = _mm512_mul_pd(v_prod, v_scale_factor);
            _mm512_store_pd(parent_clv + i, v_prod);
          }
          parent_scaler[n * rate_cats + k] += 1;
        }
      } else
        scale_mask = scale_mask & rate_mask;

      /* reset pointers to point to the start of the next p-matrix, as the
         vectorization assumes a square states_padded * states_padded matrix,
         even though the real matrix is states * states_padded */
      lmat -= displacement;
      rmat -= displacement;

      parent_clv += states_padded;
      left_clv += states_padded;
      right_clv += states_padded;
    }

    /* if *all* entries of the site CLV were below the threshold then scale
       (all) entries by PLL_SCALE_FACTOR */
    if (scale_mask == 0xF) {
      parent_clv -= span_padded;
      for (i = 0; i < span_padded; i += ELEM_PER_AVX515_REGISTER) {
        __m512d v_prod = _mm512_load_pd(parent_clv + i);
        v_prod = _mm512_mul_pd(v_prod, v_scale_factor);
        _mm512_store_pd(parent_clv + i, v_prod);
      }
      parent_clv += span_padded;
      parent_scaler[n] += 1;
    }
  }
}

PLL_EXPORT
void pll_core_update_partial_ti_20x20_avx512f(unsigned int sites,
                                              unsigned int rate_cats,
                                              double *parent_clv,
                                              unsigned int *parent_scaler,
                                              const unsigned char *left_tipchar,
                                              const double *right_clv,
                                              const double *left_matrix,
                                              const double *right_matrix,
                                              const unsigned int *right_scaler,
                                              const unsigned int *tipmap,
                                              unsigned int tipmap_size,
                                              unsigned int attrib) {
  assert(0);
//  unsigned int states = 20;
//  unsigned int states_padded = states;
//  unsigned int maxstates = tipmap_size;
//  unsigned int i, j, k, n, m;
//
//  const double *lmat;
//  const double *rmat;
//
//  unsigned int span_padded = states_padded * rate_cats;
//  unsigned int lstate;
//
//  __m256d xmm0, xmm1, xmm2, xmm3;
//
//  /* precompute a lookup table of four values per entry (one for each state),
//     for all 16 states (including ambiguities) and for each rate category. */
//  double *lookup = pll_aligned_alloc(maxstates * span_padded * sizeof(double),
//                                     PLL_ALIGNMENT_AVX);
//  if (!lookup) {
//    /* TODO: in the highly unlikely event that allocation fails, we should
//       resort to a non-lookup-precomputation version of this function,
//       available at commit e.g.  a4fc873fdc65741e402cdc1c59919375143d97d1 */
//    pll_errno = PLL_ERROR_MEM_ALLOC;
//    snprintf(pll_errmsg, 200, "Cannot allocate space for precomputation.");
//    return;
//  }
//
//  double *ptr = lookup;
//
//  /* precompute left-side values and store them in lookup table */
//  for (j = 0; j < maxstates; ++j) {
//    lmat = left_matrix;
//
//    unsigned int state = tipmap[j];
//
//    int ss = __builtin_popcount(state) == 1 ? __builtin_ctz(state) : -1;
//
//    for (n = 0; n < rate_cats; ++n) {
//      for (i = 0; i < states; ++i) {
//        double terml;
//        if (ss != -1) {
//          /* special case for non-ambiguous states */
//          terml = lmat[ss];
//        } else {
//          terml = 0;
//          for (m = 0; m < states; ++m) {
//            if ((state >> m) & 1) {
//              terml += lmat[m];
//            }
//          }
//        }
//
//        lmat += states;
//
//        ptr[i] = terml;
//      }
//
//      ptr += states;
//    }
//  }
//
//  /* scaling-related stuff */
//  unsigned int scale_mode;  /* 0 = none, 1 = per-site, 2 = per-rate */
//  unsigned int scale_mask;
//  unsigned int init_mask;
//  __m256d v_scale_threshold = _mm256_set1_pd(PLL_SCALE_THRESHOLD);
//  __m256d v_scale_factor = _mm256_set1_pd(PLL_SCALE_FACTOR);
//
//  if (!parent_scaler) {
//    /* scaling disabled / not required */
//    scale_mode = init_mask = 0;
//  } else {
//    /* determine the scaling mode and init the vars accordingly */
//    scale_mode = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 2 : 1;
//    init_mask = (scale_mode == 1) ? 0xF : 0;
//    const size_t scaler_size = (scale_mode == 2) ? sites * rate_cats : sites;
//    /* add up the scale vector of the two children if available */
//    fill_parent_scaler(scaler_size, parent_scaler, NULL, right_scaler);
//  }
//
//  size_t displacement = (states_padded - states) * (states_padded);
//
//  /* iterate over sites and compute CLV entries */
//  for (n = 0; n < sites; ++n) {
//    rmat = right_matrix;
//
//    scale_mask = init_mask;
//
//    lstate = (unsigned int) left_tipchar[n];
//
//    unsigned int loffset = lstate * span_padded;
//
//    for (k = 0; k < rate_cats; ++k) {
//      unsigned int rate_mask = 0xF;
//
//      /* iterate over quadruples of rows */
//      for (i = 0; i < states_padded; i += 4) {
//        __m256d v_termb0 = _mm256_setzero_pd();
//        __m256d v_termb1 = _mm256_setzero_pd();
//        __m256d v_termb2 = _mm256_setzero_pd();
//        __m256d v_termb3 = _mm256_setzero_pd();
//
//        __m256d v_mat;
//        __m256d v_rclv;
//
//        /* point to the four rows of the right matrix */
//        const double *rm0 = rmat;
//        const double *rm1 = rm0 + states_padded;
//        const double *rm2 = rm1 + states_padded;
//        const double *rm3 = rm2 + states_padded;
//
//        /* iterate over quadruples of columns */
//        for (j = 0; j < states_padded; j += 4) {
//          v_rclv = _mm256_load_pd(right_clv + j);
//
//          /* row 0 */
//          v_mat = _mm256_load_pd(rm0);
//          v_termb0 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb0);
//          rm0 += 4;
//
//          /* row 1 */
//          v_mat = _mm256_load_pd(rm1);
//          v_termb1 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb1);
//          rm1 += 4;
//
//          /* row 2 */
//          v_mat = _mm256_load_pd(rm2);
//          v_termb2 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb2);
//          rm2 += 4;
//
//          /* row 3 */
//          v_mat = _mm256_load_pd(rm3);
//          v_termb3 = _mm256_fmadd_pd(v_mat, v_rclv, v_termb3);
//          rm3 += 4;
//        }
//
//        /* point pmatrix to the next four rows */
//        rmat = rm3;
//
//        /* load x from precomputed lookup table */
//        __m256d v_terma_sum = _mm256_load_pd(lookup + loffset);
//        loffset += 4;
//
//        /* compute termb */
//        xmm0 = _mm256_unpackhi_pd(v_termb0, v_termb1);
//        xmm1 = _mm256_unpacklo_pd(v_termb0, v_termb1);
//
//        xmm2 = _mm256_unpackhi_pd(v_termb2, v_termb3);
//        xmm3 = _mm256_unpacklo_pd(v_termb2, v_termb3);
//
//        xmm0 = _mm256_add_pd(xmm0, xmm1);
//        xmm1 = _mm256_add_pd(xmm2, xmm3);
//
//        xmm2 = _mm256_permute2f128_pd(xmm0, xmm1, _MM_SHUFFLE(0, 2, 0, 1));
//
//        xmm3 = _mm256_blend_pd(xmm0, xmm1, 12);
//
//        __m256d v_termb_sum = _mm256_add_pd(xmm2, xmm3);
//
//        __m256d v_prod = _mm256_mul_pd(v_terma_sum, v_termb_sum);
//
//        /* check if scaling is needed for the current rate category */
//        __m256d v_cmp = _mm256_cmp_pd(v_prod, v_scale_threshold, _CMP_LT_OS);
//        rate_mask = rate_mask & _mm256_movemask_pd(v_cmp);
//
//        _mm256_store_pd(parent_clv + i, v_prod);
//      }
//
//      if (scale_mode == 2) {
//        /* PER-RATE SCALING: if *all* entries of the *rate* CLV were below
//         * the threshold then scale (all) entries by PLL_SCALE_FACTOR */
//        if (rate_mask == 0xF) {
//          for (i = 0; i < states_padded; i += 4) {
//            __m256d v_prod = _mm256_load_pd(parent_clv + i);
//            v_prod = _mm256_mul_pd(v_prod, v_scale_factor);
//            _mm256_store_pd(parent_clv + i, v_prod);
//          }
//          parent_scaler[n * rate_cats + k] += 1;
//        }
//      } else
//        scale_mask = scale_mask & rate_mask;
//
//      /* reset pointers to point to the start of the next p-matrix, as the
//         vectorization assumes a square states_padded * states_padded matrix,
//         even though the real matrix is states * states_padded */
//      rmat -= displacement;
//
//      parent_clv += states_padded;
//      right_clv += states_padded;
//    }
//
//    /* if *all* entries of the site CLV were below the threshold then scale
//       (all) entries by PLL_SCALE_FACTOR */
//    if (scale_mask == 0xF) {
//      parent_clv -= span_padded;
//      for (i = 0; i < span_padded; i += 4) {
//        __m256d v_prod = _mm256_load_pd(parent_clv + i);
//        v_prod = _mm256_mul_pd(v_prod, v_scale_factor);
//        _mm256_store_pd(parent_clv + i, v_prod);
//      }
//      parent_clv += span_padded;
//      parent_scaler[n] += 1;
//    }
//  }
//  pll_aligned_free(lookup);
}

PLL_EXPORT void pll_core_update_partial_ii_avx512f(unsigned int states,
                                                   unsigned int sites,
                                                   unsigned int rate_cats,
                                                   double *parent_clv,
                                                   unsigned int *parent_scaler,
                                                   const double *left_clv,
                                                   const double *right_clv,
                                                   const double *left_matrix,
                                                   const double *right_matrix,
                                                   const unsigned int *left_scaler,
                                                   const unsigned int *right_scaler,
                                                   unsigned int attrib) {
  unsigned int i, j, k, n;

  const double *lmat;
  const double *rmat;

  unsigned int states_padded = (states + 7) & (0xFFFFFFFF - 7);
  unsigned int span_padded = states_padded * rate_cats;

  /* dedicated functions for 4x4 matrices */
  if (states == 4) {
    /* TODO: Implement avx512 4x4 case */
    pll_core_update_partial_ii_4x4_avx(sites,
                                       rate_cats,
                                       parent_clv,
                                       parent_scaler,
                                       left_clv,
                                       right_clv,
                                       left_matrix,
                                       right_matrix,
                                       left_scaler,
                                       right_scaler,
                                       attrib);
    return;
  }
  // dedicated function for 20x20 matrices
  if (states == 20) {
    pll_core_update_partial_ii_20x20_avx512f(sites,
                                             rate_cats,
                                             parent_clv,
                                             parent_scaler,
                                             left_clv,
                                             right_clv,
                                             left_matrix,
                                             right_matrix,
                                             left_scaler,
                                             right_scaler,
                                             attrib);
    return;
  }

  /* scaling-related stuff */
  unsigned int scale_mode;  /* 0 = none, 1 = per-site, 2 = per-rate */
  unsigned int scale_mask;
  unsigned int init_mask;
  __m512d v_scale_threshold = _mm512_set1_pd(PLL_SCALE_THRESHOLD);
  __m512d v_scale_factor = _mm512_set1_pd(PLL_SCALE_FACTOR);

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

  if (!parent_scaler) {
    /* scaling disabled / not required */
    scale_mode = init_mask = 0;
  } else {
    /* determine the scaling mode and init the vars accordingly */
    scale_mode = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 2 : 1;
    init_mask = (scale_mode == 1) ? 0xF : 0;
    const size_t scaler_size = (scale_mode == 2) ? sites * rate_cats : sites;
    /* add up the scale vector of the two children if available */
    fill_parent_scaler(scaler_size, parent_scaler, left_scaler, right_scaler);
  }

  size_t displacement = (states_padded - states) * (states_padded);

  /* compute CLV */
  for (n = 0; n < sites; ++n) {
    lmat = left_matrix;
    rmat = right_matrix;
    scale_mask = init_mask;

    for (k = 0; k < rate_cats; ++k) {
      __mmask8 rate_mask = 0xF;

      /* iterate over octuple of rows */
      for (i = 0; i < states_padded; i += ELEM_PER_AVX515_REGISTER) {
        __m512d v_terma0 = _mm512_setzero_pd();
        __m512d v_termb0 = _mm512_setzero_pd();
        __m512d v_terma1 = _mm512_setzero_pd();
        __m512d v_termb1 = _mm512_setzero_pd();
        __m512d v_terma2 = _mm512_setzero_pd();
        __m512d v_termb2 = _mm512_setzero_pd();
        __m512d v_terma3 = _mm512_setzero_pd();
        __m512d v_termb3 = _mm512_setzero_pd();
        __m512d v_terma4 = _mm512_setzero_pd();
        __m512d v_termb4 = _mm512_setzero_pd();
        __m512d v_terma5 = _mm512_setzero_pd();
        __m512d v_termb5 = _mm512_setzero_pd();
        __m512d v_terma6 = _mm512_setzero_pd();
        __m512d v_termb6 = _mm512_setzero_pd();
        __m512d v_terma7 = _mm512_setzero_pd();
        __m512d v_termb7 = _mm512_setzero_pd();

        __m512d v_mat;
        __m512d v_lclv;
        __m512d v_rclv;

        /* point to the four rows of the left matrix */
        const double *lm0 = lmat;
        const double *lm1 = lm0 + states_padded;
        const double *lm2 = lm1 + states_padded;
        const double *lm3 = lm2 + states_padded;
        const double *lm4 = lm3 + states_padded;
        const double *lm5 = lm4 + states_padded;
        const double *lm6 = lm5 + states_padded;
        const double *lm7 = lm6 + states_padded;

        /* point to the four rows of the right matrix */
        const double *rm0 = rmat;
        const double *rm1 = rm0 + states_padded;
        const double *rm2 = rm1 + states_padded;
        const double *rm3 = rm2 + states_padded;
        const double *rm4 = rm3 + states_padded;
        const double *rm5 = rm4 + states_padded;
        const double *rm6 = rm5 + states_padded;
        const double *rm7 = rm6 + states_padded;

        /* iterate over octuple of columns */
        for (j = 0; j < states_padded; j += ELEM_PER_AVX515_REGISTER) {
          v_lclv = _mm512_load_pd(left_clv + j);
          v_rclv = _mm512_load_pd(right_clv + j);

          /* row 0 */
          v_mat = _mm512_load_pd(lm0);
          v_terma0 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma0);

          v_mat = _mm512_load_pd(rm0);
          v_termb0 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb0);
          lm0 += ELEM_PER_AVX515_REGISTER;
          rm0 += ELEM_PER_AVX515_REGISTER;

          /* row 1 */
          v_mat = _mm512_load_pd(lm1);
          v_terma1 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma1);

          v_mat = _mm512_load_pd(rm1);
          v_termb1 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb1);
          lm1 += ELEM_PER_AVX515_REGISTER;
          rm1 += ELEM_PER_AVX515_REGISTER;

          /* row 2 */
          v_mat = _mm512_load_pd(lm2);
          v_terma2 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma2);

          v_mat = _mm512_load_pd(rm2);
          v_termb2 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb2);
          lm2 += ELEM_PER_AVX515_REGISTER;
          rm2 += ELEM_PER_AVX515_REGISTER;

          /* row 3 */
          v_mat = _mm512_load_pd(lm3);
          v_terma3 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma3);

          v_mat = _mm512_load_pd(rm3);
          v_termb3 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb3);

          lm3 += ELEM_PER_AVX515_REGISTER;
        }

        if (i < 16) {
          for (j = 0; j < states_padded; j += ELEM_PER_AVX515_REGISTER) {

            /* row 4 */
            v_mat = _mm512_load_pd(lm4);
            v_terma4 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma4);

            v_mat = _mm512_load_pd(rm4);
            v_termb4 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb4);

            lm4 += ELEM_PER_AVX515_REGISTER;
            rm4 += ELEM_PER_AVX515_REGISTER;

            /* row 5 */
            v_mat = _mm512_load_pd(lm5);
            v_terma5 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma5);

            v_mat = _mm512_load_pd(rm5);
            v_termb5 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb5);

            lm5 += ELEM_PER_AVX515_REGISTER;
            rm5 += ELEM_PER_AVX515_REGISTER;

            /* row 6 */
            v_mat = _mm512_load_pd(lm6);
            v_terma6 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma6);

            v_mat = _mm512_load_pd(rm6);
            v_termb6 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb6);

            lm6 += ELEM_PER_AVX515_REGISTER;
            rm6 += ELEM_PER_AVX515_REGISTER;

            /* row 7 */
            v_mat = _mm512_load_pd(lm7);
            v_terma7 = _mm512_fmadd_pd(v_mat, v_lclv, v_terma7);

            v_mat = _mm512_load_pd(rm7);
            v_termb7 = _mm512_fmadd_pd(v_mat, v_rclv, v_termb7);

            lm7 += ELEM_PER_AVX515_REGISTER;
            rm7 += ELEM_PER_AVX515_REGISTER;
          }

        } else {
          lm7 += states_padded;
          rm7 += states_padded;
        }

        /* point pmatrix to the next four rows */
        lmat = lm7;
        rmat = rm7;

        __m512d ymm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma0, v_terma1),
                                     _mm512_unpacklo_pd(v_terma0, v_terma1));
        __m512d ymm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma2, v_terma3),
                                     _mm512_unpacklo_pd(v_terma2, v_terma3));
        __m512d ymm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma4, v_terma5),
                                     _mm512_unpacklo_pd(v_terma4, v_terma5));
        __m512d ymm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_terma6, v_terma7),
                                     _mm512_unpacklo_pd(v_terma6, v_terma7));

        __m512d zmm0 = _mm512_add_pd(_mm512_permutex2var_pd(ymm0, permute_mask, ymm2),
                                     _mm512_mask_blend_pd(0xF0, ymm0, ymm2));

        __m512d zmm1 = _mm512_add_pd(_mm512_permutex2var_pd(ymm1, permute_mask, ymm3),
                                     _mm512_mask_blend_pd(0xF0, ymm1, ymm3));

        __m512d v_terma_sum = _mm512_add_pd(_mm512_permutex2var_pd(zmm0,
                                                                   permute_mask_final_stage,
                                                                   zmm1),
                                            _mm512_mask_blend_pd(0xCC, zmm0, zmm1));

        /* compute termb */

        ymm0 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb0, v_termb1),
                             _mm512_unpacklo_pd(v_termb0, v_termb1));
        ymm1 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb2, v_termb3),
                             _mm512_unpacklo_pd(v_termb2, v_termb3));
        ymm2 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb4, v_termb5),
                             _mm512_unpacklo_pd(v_termb4, v_termb5));
        ymm3 = _mm512_add_pd(_mm512_unpackhi_pd(v_termb6, v_termb7),
                             _mm512_unpacklo_pd(v_termb6, v_termb7));

        zmm0 = _mm512_add_pd(_mm512_permutex2var_pd(ymm0, permute_mask, ymm2),
                             _mm512_mask_blend_pd(0xF0, ymm0, ymm2));

        zmm1 = _mm512_add_pd(_mm512_permutex2var_pd(ymm1, permute_mask, ymm3),
                             _mm512_mask_blend_pd(0xF0, ymm1, ymm3));

        __m512d v_termb_sum = _mm512_add_pd(_mm512_permutex2var_pd(zmm0,
                                                                   permute_mask_final_stage,
                                                                   zmm1),
                                            _mm512_mask_blend_pd(0xCC, zmm0, zmm1));

        __m512d v_prod = _mm512_mul_pd(v_terma_sum, v_termb_sum);

        /* check if scaling is needed for the current rate category */
        rate_mask = _mm512_cmp_pd_mask(v_prod, v_scale_threshold, _CMP_LT_OS);

        _mm512_store_pd(parent_clv + i, v_prod);
      }

      if (scale_mode == 2) {
        /* PER-RATE SCALING: if *all* entries of the *rate* CLV were below
       * the threshold then scale (all) entries by PLL_SCALE_FACTOR */
        if (rate_mask == 0xF) {
          for (i = 0; i < states_padded; i += ELEM_PER_AVX515_REGISTER) {
            __m512d v_prod = _mm512_load_pd(parent_clv + i);
            v_prod = _mm512_mul_pd(v_prod, v_scale_factor);
            _mm512_store_pd(parent_clv + i, v_prod);
          }
          parent_scaler[n * rate_cats + k] += 1;
        }
      } else
        scale_mask = scale_mask & rate_mask;

      /* reset pointers to point to the start of the next p-matrix, as the
         vectorization assumes a square states_padded * states_padded matrix,
         even though the real matrix is states * states_padded */
      lmat -= displacement;
      rmat -= displacement;

      parent_clv += states_padded;
      left_clv += states_padded;
      right_clv += states_padded;
    }

    /* if *all* entries of the site CLV were below the threshold then scale
       (all) entries by PLL_SCALE_FACTOR */
    if (scale_mask == 0xF) {
      parent_clv -= span_padded;
      for (i = 0; i < span_padded; i += ELEM_PER_AVX515_REGISTER) {
        __m512d v_prod = _mm512_load_pd(parent_clv + i);
        v_prod = _mm512_mul_pd(v_prod, v_scale_factor);
        _mm512_store_pd(parent_clv + i, v_prod);
      }
      parent_clv += span_padded;
      parent_scaler[n] += 1;
    }
  }
}
