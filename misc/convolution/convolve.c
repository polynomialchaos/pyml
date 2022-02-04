#include <stdio.h>
#include <stdlib.h>

void convolve_2d(
    const size_t *s_i, double *input,
    const size_t *s_k, double *kernel,
    const size_t *s_o, double *output)
{
    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i < s_o[0]; ++i)
    {
        for (int j = 0; j < s_o[1]; ++j)
        {
            output[i * s_o[1] + j] = 0;

            for (int k = 0; k < s_k[0]; ++k)
            {
                for (int l = 0; l < s_k[1]; ++l)
                {
                    output[i * s_o[1] + j] +=
                        input[(i + k) * s_i[1] + j + l] *
                        kernel[k * s_k[1] + l];
                }
            }
        }
    }
}