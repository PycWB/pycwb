// gcc -c watasm.S

.globl _watasm_data
.globl _watasm_filter
.globl _watasm_xmm0

.globl _sse_dp4
.globl _sse_dp5
.globl _sse_dp6
.globl _sse_dp7
.globl _sse_dp8
.globl _sse_dp9
.globl _sse_dp10
.globl _sse_dp11

.data
_watasm_data: .quad 0
_watasm_filter: .quad 0
.align 4
_watasm_xmm0: .quad 0, 0

.text

_sse_dp4:
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      
      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      addps %xmm2, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip) 
      ret

_sse_dp5:  
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      movaps 64(%rdi), %xmm4  
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      mulps 64(%rsi), %xmm4

      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      
      addps %xmm4, %xmm0
      addps %xmm2, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip)    
      ret


_sse_dp6:  
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      movaps 64(%rdi), %xmm4  
      movaps 80(%rdi), %xmm5
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      mulps 64(%rsi), %xmm4
      mulps 80(%rsi), %xmm5

      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      addps %xmm5, %xmm4
      
      addps %xmm4, %xmm0
      addps %xmm2, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip)    
      ret

_sse_dp7:  
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      movaps 64(%rdi), %xmm4  
      movaps 80(%rdi), %xmm5
      movaps 96(%rdi), %xmm6
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      mulps 64(%rsi), %xmm4
      mulps 80(%rsi), %xmm5
      mulps 96(%rsi), %xmm6

      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      addps %xmm5, %xmm4
      
      addps %xmm2, %xmm0
      addps %xmm6, %xmm4
      
      addps %xmm4, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip)    
      ret


_sse_dp8:  
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      movaps 64(%rdi), %xmm4  
      movaps 80(%rdi), %xmm5
      movaps 96(%rdi), %xmm6
      movaps 112(%rdi), %xmm7
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      mulps 64(%rsi), %xmm4
      mulps 80(%rsi), %xmm5
      mulps 96(%rsi), %xmm6
      mulps 112(%rsi), %xmm7

      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      addps %xmm5, %xmm4
      addps %xmm7, %xmm6
      
      addps %xmm2, %xmm0
      addps %xmm6, %xmm4
      
      addps %xmm4, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip)    
      ret


_sse_dp9:  
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      movaps 64(%rdi), %xmm4  
      movaps 80(%rdi), %xmm5
      movaps 96(%rdi), %xmm6
      movaps 112(%rdi), %xmm7
      movaps 128(%rdi), %xmm8
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      mulps 64(%rsi), %xmm4
      mulps 80(%rsi), %xmm5
      mulps 96(%rsi), %xmm6
      mulps 112(%rsi), %xmm7
      mulps 128(%rsi), %xmm8

      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      addps %xmm5, %xmm4
      addps %xmm7, %xmm6
      
      addps %xmm2, %xmm0
      addps %xmm6, %xmm4
      
      addps %xmm4, %xmm0
      addps %xmm8, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip)    
      ret

_sse_dp10:  
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      movaps 64(%rdi), %xmm4  
      movaps 80(%rdi), %xmm5
      movaps 96(%rdi), %xmm6
      movaps 112(%rdi), %xmm7
      movaps 128(%rdi), %xmm8
      movaps 144(%rdi), %xmm9
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      mulps 64(%rsi), %xmm4
      mulps 80(%rsi), %xmm5
      mulps 96(%rsi), %xmm6
      mulps 112(%rsi), %xmm7
      mulps 128(%rsi), %xmm8
      mulps 144(%rsi), %xmm9

      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      addps %xmm5, %xmm4
      addps %xmm7, %xmm6
      addps %xmm9, %xmm8
      
      addps %xmm2, %xmm0
      addps %xmm6, %xmm4
      
      addps %xmm4, %xmm0
      addps %xmm8, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip)    
      ret

_sse_dp11:  
      mov _watasm_data(%rip), %rdi
      mov _watasm_filter(%rip), %rsi
      
      movaps (%rdi), %xmm0
      movaps 16(%rdi), %xmm1
      movaps 32(%rdi), %xmm2
      movaps 48(%rdi), %xmm3
      movaps 64(%rdi), %xmm4  
      movaps 80(%rdi), %xmm5
      movaps 96(%rdi), %xmm6
      movaps 112(%rdi), %xmm7
      movaps 128(%rdi), %xmm8
      movaps 144(%rdi), %xmm9
      movaps 160(%rdi), %xmm10
      
      mulps (%rsi), %xmm0
      mulps 16(%rsi), %xmm1
      mulps 32(%rsi), %xmm2
      mulps 48(%rsi), %xmm3
      mulps 64(%rsi), %xmm4
      mulps 80(%rsi), %xmm5
      mulps 96(%rsi), %xmm6
      mulps 112(%rsi), %xmm7
      mulps 128(%rsi), %xmm8
      mulps 144(%rsi), %xmm9
      mulps 160(%rsi), %xmm10
      
      addps %xmm1, %xmm0
      addps %xmm3, %xmm2
      addps %xmm5, %xmm4
      addps %xmm7, %xmm6
      addps %xmm9, %xmm8
      
      addps %xmm2, %xmm0
      addps %xmm6, %xmm4
      addps %xmm10, %xmm8
      
      addps %xmm4, %xmm0
      addps %xmm8, %xmm0
      
      movaps %xmm0, _watasm_xmm0(%rip)    
      ret

