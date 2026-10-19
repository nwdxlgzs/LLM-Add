/* The same probe is compiled as native C and WASM; no JS arithmetic kernels. */
#include "add_column.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif
static int dump_logits(cn_model *m,const char *path,int fields) {
    FILE *f=fopen(path,"wb");cn_context *s=cn_context_create(m,0);
    if(!f||!s){if(f)fclose(f);cn_context_free(s);return 0;}
    for(int k=0;k<4;k++) {
        int t[ADD_CHAT_CAPACITY],p,n;
        if(k<2)n=add_column_example(k?9:7,k?9:1,k?1:0,0,t,ADD_CHAT_CAPACITY,&p);
        else n=add_chat_example_format(k==2?"99":"1234567890",k==2?"1":"9876543210",k==2||fields,0,t,ADD_CHAT_CAPACITY,&p,fields);
        cn_context_reset(s);const float *x=NULL;
        for(int j=0;j<n-1;j++) {
            if(!cn_forward(s,t[j],&x)||fwrite(x,sizeof(float),CN_VOCAB,f)!=CN_VOCAB){fclose(f);cn_context_free(s);return 0;}
        }
    }
    int ok=!ferror(f);if(fclose(f))ok=0;cn_context_free(s);return ok;
}
EMSCRIPTEN_KEEPALIVE int parity_run(const char *input,const char *prefix,int mode,int steps,int backend) {
    /* Modes 10..16 explicitly select labeled whole-dialogue format; historical
       modes retain compact targets, independently of the checkpoint name. */
    int fields=mode>=10;mode%=10;
    char error[256],path[1200];cn_model *m=cn_load(input,error,sizeof(error)),*ref=NULL;
    if(!m||!cn_model_set_backend(m,(cn_cpu_backend)backend))return 0;
    snprintf(path,sizeof(path),"%s-before.bin",prefix);if(!dump_logits(m,path,fields))goto fail;
    cn_optimizer opt=cn_optimizer_default();opt.learning_rate=.0001f;opt.beta2=.99f;opt.weight_decay=.001f;
    uint64_t rng=1234;double reward=0;int informative=0,signals=0;cn_metrics stats={0};
    if(mode==3||mode==5||mode==7){ref=cn_load(input,error,sizeof(error));if(!ref||!cn_model_set_backend(ref,(cn_cpu_backend)backend))goto fail;}
    if((mode==4||mode==6)&&!cn_model_set_qat(m,CN_MXFP4))goto fail;
    snprintf(path,sizeof(path),"%s-steps.json",prefix);FILE *log=fopen(path,"wb");if(!log)goto fail;
    fputc('[',log);
    for(int i=0;i<steps;i++) {
        int ok=(mode==1||mode==5||mode==6||mode==7)?add_chat_update_format(m,ref,mode==7?7:mode==5?5:mode==6?6:4,i+1,steps,2,1,&rng,&opt,&stats,&reward,&informative,fields)
                      :add_column_update(m,ref,mode==3?5:4,i+1,steps,2,&rng,&opt,&stats,&reward,&informative);
        if(!ok){fclose(log);goto fail;}
        signals+=informative;
        fprintf(log,"%s{\"loss\":%.9g,\"grad_norm\":%.9g,\"step\":%llu,\"reward\":%.9g,\"informative\":%d}",i?",":"",stats.loss,stats.grad_norm,(unsigned long long)stats.step,reward,informative);
    }
    fputc(']',log);if(fclose(log))goto fail;
    snprintf(path,sizeof(path),"%s-after.bin",prefix);if(!dump_logits(m,path,fields))goto fail;
    snprintf(path,sizeof(path),"%s-model.cn",prefix);if(!cn_save(m,path,1))goto fail;
    cn_model_free(m);cn_model_free(ref);return mode==3&&steps>0&&!signals?2:1;
fail:
    cn_model_free(m);cn_model_free(ref);return 0;
}
#ifndef __EMSCRIPTEN__
int main(int argc,char **argv) {
    if(argc==2){cn_config c;add_chat_config(&c,20261018);char err[256];cn_model*m=cn_model_create(&c,err,sizeof(err));if(!m)return 1;int ok=cn_save(m,argv[1],1);cn_model_free(m);return !ok;}
    if(argc!=6)return 2;
    int ok=parity_run(argv[1],argv[2],atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
    return ok==1?0:ok==2?3:1;
}
#endif
