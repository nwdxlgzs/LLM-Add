#include "add_column.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
static int sum_model(cn_context *s,const char *a,const char *b,char *out) {
    size_t na=strlen(a),nb=strlen(b),n=na>nb?na:nb,w=0;int carry=0;
    for(size_t pos=0;pos<n;pos++) {
        int ids[5],da=pos<na?a[na-1-pos]-'0':0,db=pos<nb?b[nb-1-pos]-'0':0;
        if(!add_column_local(s,da,db,carry,ids,NULL,NULL))return 0;
        out[w++]=(char)ids[0];carry=ids[2]-'0';
    }
    if(carry)out[w++]=(char)('0'+carry);
    while(w>1&&out[w-1]=='0')w--;
    for(size_t j=0;j<w/2;j++){char c=out[j];out[j]=out[w-1-j];out[w-1-j]=c;}out[w]=0;return 1;
}
int main(int argc,char **argv) {
    if(argc!=4){fprintf(stderr,"column_eval MODEL REPORT RANDOM_COUNT\n");return 2;}
    char error[256];cn_model *m=cn_load(argv[1],error,sizeof(error));if(!m)return 1;
    FILE *f=fopen(argv[2],"wb");if(!f){cn_model_free(m);return 1;}
    fprintf(f,"{\"model\":\"%s\",\"parameters\":%zu,\"precision\":%d,\"backends\":[",argv[1],cn_parameter_count(m),cn_model_precision(m));
    int all=1,first=1;
    for(int b=CN_CPU_SCALAR;b<CN_CPU_BACKENDS;b++)if(cn_cpu_backend_available((cn_cpu_backend)b)){
        add_chat_validation r;
        if(!cn_model_set_backend(m,(cn_cpu_backend)b)||!add_column_validate(m,&r))return 1;
        fprintf(f,"%s{\"name\":\"%s\",\"correct\":%d,\"total\":200,\"min_margin\":%.9g,\"loss\":%.9g}",first?"":",",cn_cpu_backend_name((cn_cpu_backend)b),r.exact.correct,r.exact.margin,r.loss);first=0;all &= r.exact.correct==200;
    }
    cn_model_set_backend(m,CN_CPU_AUTO);cn_context *s=cn_context_create(m,0);if(!s)return 1;
    int counts[5]={0},totals[5]={0},widths[5]={1,2,4,6,10};uint64_t rng=20261022;
    int repeat=atoi(argv[3]);if(repeat<0||repeat>100000)return 2;
    for(int level=0;level<5;level++)for(int i=0;i<repeat;i++){
        uint64_t power=1;for(int j=0;j<widths[level];j++)power*=10;
        uint64_t a=add_random(&rng)%power,b=add_random(&rng)%power;
        if(i%16==0){a=power-1;b=1;}
        char as[24],bs[24],answer[32],expected[32];
        snprintf(as,sizeof(as),"%" PRIu64,a);snprintf(bs,sizeof(bs),"%" PRIu64,b);snprintf(expected,sizeof(expected),"%" PRIu64,a+b);
        totals[level]++;counts[level]+=sum_model(s,as,bs,answer)&&!strcmp(answer,expected);
    }
    fputs("],\"random_tests\":[",f);
    for(int i=0;i<5;i++){fprintf(f,"%s{\"width\":%d,\"correct\":%d,\"total\":%d}",i?",":"",widths[i],counts[i],totals[i]);all &= counts[i]==totals[i];}
    char *longa=malloc(1001),*answer=malloc(1003);if(!longa||!answer)return 1;
    memset(longa,'9',1000);longa[1000]=0;
    int longok=sum_model(s,longa,"1",answer)&&strlen(answer)==1001&&answer[0]=='1';
    for(int i=1;i<=1000&&longok;i++)longok &= answer[i]=='0';
    fprintf(f,"],\"carry_1000_digits\":%s,\"passed\":%s}\n",longok?"true":"false",all&&longok?"true":"false");
    free(longa);free(answer);cn_context_free(s);cn_model_free(m);if(fclose(f))return 1;return all&&longok?0:1;
}
