#include "quant_kernel.h"
#include "bit_helper.cu"

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        int man_bits,
                                        int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  unsigned int overflows=0;
  unsigned int underflows=0;
  unsigned int total=size;

  if (index < size) {
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int target,quantize_bits,quantize_bits_tmp;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      underflows=underflows+1;
      float shift_float,val;
      int shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val=a[index]+shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else{
      quantize_bits_tmp = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits_tmp);
      quantized = BITS_TO_FLOAT(&quantize_bits);
      if (quantize_bits!=quantize_bits_tmp){
        overflows=overflows+1;
      }
    }
    o[index] = quantized;
  }
  
  // Log overflows if necessary
  char* QPYTORCH_LOG = getenv("QPYTORCH_LOG");
  if (strcmp(QPYTORCH_LOG,"ALL") == 0){
    char* SLURM_JOB_ID = getenv("SLURM_JOB_ID");
    char file_name[80] = "QPYTORCH_LOG_";
    if (SLURM_JOB_ID!=NULL)
      strcat(file_name, SLURM_JOB_ID);
    strcat(file_name, ".txt");
    FILE *f;
    f = fopen(file_name, "a");
    fprintf(f, "%d %d %d\n",total,overflows,underflows);
    fclose(f);
  }

}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     int man_bits,
                                     int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int overflows=0;
  unsigned int underflows=0;
  unsigned int total=size;

  if (index < size) {
    unsigned int target,quantize_bits,quantize_bits_tmp;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      underflows=underflows+1;
      float shift_float,val;
      int shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val=a[index]+shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else{
      quantize_bits_tmp = round_bitwise_nearest(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits_tmp);
      quantized = BITS_TO_FLOAT(&quantize_bits);
      if (quantize_bits!=quantize_bits_tmp){
        overflows=overflows+1;
      }
    }
    o[index] = quantized;
  }

  // Log overflows if necessary
  char* QPYTORCH_LOG = getenv("QPYTORCH_LOG");
  if (strcmp(QPYTORCH_LOG,"ALL") == 0){
    char* SLURM_JOB_ID = getenv("SLURM_JOB_ID");
    char file_name[80] = "QPYTORCH_LOG_";
    if (SLURM_JOB_ID!=NULL)
      strcat(file_name, SLURM_JOB_ID);
    strcat(file_name, ".txt");
    FILE *f;
    f = fopen(file_name, "a");
    fprintf(f, "%d %d %d\n",total,overflows,underflows);
    fclose(f);
  }

}
