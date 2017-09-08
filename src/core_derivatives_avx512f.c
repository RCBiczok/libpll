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

  /* dedicated functions for 4x4 matrices */
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
  }

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

  unsigned int states_padded = (states + 7) & (0xFFFFFFFF - 7);

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
        /* point to the four rows of the eigenvecs matrix */
        const double *em0 = c_eigenvecs;
        const double *em1 = em0 + states_padded;
        const double *em2 = em1 + states_padded;
        const double *em3 = em2 + states_padded;
        const double *em4 = em3 + states_padded;
        const double *em5 = em4 + states_padded;
        const double *em6 = em5 + states_padded;
        const double *em7 = em6 + states_padded;
        c_eigenvecs += ELEM_PER_AVX515_REGISTER * states_padded;

        /* point to the four rows of the inv_eigenvecs matrix */
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
  //TODOs
  return PLL_SUCCESS;
}
