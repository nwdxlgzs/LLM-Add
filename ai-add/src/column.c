#include "add_column.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int column_prompt(int a, int b, int ci, int copy, int *t, int cap) {
    if (a < 0 || a > 9 || b < 0 || b > 9 || ci < 0 || ci > 1 || !t || cap < 1)
        return 0;
    char input[160];
    snprintf(input, sizeof(input), "<|im_start|>user\n%s a=%d b=%d cin=%d<|im_end|>\n<|im_start|>assistant\n",
             copy ? "copy" : "add", a, b, ci);
    size_t p = cn_encode(input, strlen(input), 1, t, (size_t)cap);
    if (p == SIZE_MAX || p >= (size_t)cap) return 0;
    return (int)p;
}
int add_column_example(int a, int b, int ci, int copy, int *t, int cap, int *prompt) {
    if(!prompt)return 0;
    int p=column_prompt(a,b,ci,copy,t,cap);
    if(!p)return 0;
    char output[80];int sum=a+b+ci;
    snprintf(output, sizeof(output), "%d %d<|im_end|><|endoftext|>", copy ? a : sum % 10, copy ? b : sum / 10);
    size_t n = cn_encode(output, strlen(output), 1, t + p, (size_t)cap - p);
    if (n == SIZE_MAX || n > (size_t)cap - p) return 0;
    *prompt = (int)p;
    return (int)(p + n);
}
static int argmax(const float *x, float *margin) {
    int top = 0;
    for (int j = 0; j < CN_VOCAB; j++) {
        if (!isfinite(x[j])) return -1;
        if (x[j] > x[top]) top = j;
    }
    float other = -FLT_MAX;
    for (int j = 0; j < CN_VOCAB; j++) if (j != top && x[j] > other) other = x[j];
    if (margin && x[top] - other < *margin) *margin = x[top] - other;
    return top;
}
int add_column_local(cn_context *s, int a, int b, int ci, int ids[5], uint64_t *rng, float *margin) {
    int t[128], prompt = column_prompt(a,b,ci,0,t,128);
    if (!s || !prompt) return 0;
    const float *x = NULL;
    cn_context_reset(s);
    for (int j = 0; j < 5; j++) ids[j] = -1;
    for (int j = 0; j < prompt; j++) if (!cn_forward(s,t[j],&x)) return 0;
    for (int j = 0; j < 5; j++) {
        ids[j] = rng ? cn_sample(x,CN_VOCAB,1.f,0,rng) : argmax(x,margin);
        if (ids[j] < 0) return 0;
        if (ids[j] == CN_EOT) break;
        if (j < 4 && !cn_forward(s,ids[j],&x)) return 0;
    }
    return ids[0] >= '0' && ids[0] <= '9' && ids[1] == ' ' &&
           ids[2] >= '0' && ids[2] <= '1' && ids[3] == CN_IM_END && ids[4] == CN_EOT;
}
int add_column_validate(cn_model *m, add_chat_validation *r) {
    if (!m || !r) return 0;
    memset(r,0,sizeof(*r)); r->exact.total=200; r->exact.margin=FLT_MAX;
    cn_context *s=cn_context_create(m,0);
    if (!s) return 0;
    for (int id=0;id<200;id++) {
        int a=id/20,b=id/2%10,ci=id%2,t[128],p=0;
        int n=add_column_example(a,b,ci,0,t,128,&p), match=1;
        const float *x=NULL; cn_context_reset(s);
        for (int j=0;j<p;j++) if(!cn_forward(s,t[j],&x)) {cn_context_free(s);return 0;}
        for(int j=p;j<n;j++) {
            int correct=argmax(x,NULL)==t[j]; match &= correct;
            r->token_correct+=correct;r->token_total++;
            float delta=x[t[j]],other=-FLT_MAX;
            double mx=x[0],sum=0;
            for(int k=0;k<CN_VOCAB;k++) {if(k!=t[j]&&x[k]>other)other=x[k];if(x[k]>mx)mx=x[k];}
            if(delta-other<r->exact.margin)r->exact.margin=delta-other;
            for(int k=0;k<CN_VOCAB;k++)sum+=exp(x[k]-mx);
            r->loss+=log(sum)+mx-x[t[j]];
            if(j<n-1&&!cn_forward(s,t[j],&x)){cn_context_free(s);return 0;}
        }
        r->exact.correct+=match;
    }
    cn_context_free(s);r->loss/=r->token_total;r->exact.probability=exp(-r->loss);return 1;
}
int add_column_update(cn_model *m,cn_model *ref,int stage,int step,int total,int batch,
                      uint64_t *rng,const cn_optimizer *opt,cn_metrics *stats,double *reward,int *informative) {
    memset(stats,0,sizeof(*stats));*reward=0;*informative=0;
    if(stage!=5) {
        for(int k=0;k<batch;k++) {
            int id=(int)(add_random(rng)%200),a=id/20,b=id/2%10,ci=id%2;
            if(stage==2){ci=0;b%=10-a;} else if(stage==3)ci=0;
            int t[128],p=0,n=add_column_example(a,b,ci,stage==1,t,128,&p);
            if(!n)return 0;
            cn_token_objective obj[128]={{0}};
            for(int j=p;j<n;j++)obj[j].ce=1.f/((n-p)*batch);
            cn_metrics one;
            if(!cn_objective_backward(m,t,(size_t)n,obj,k!=0,1.f/batch,&one))return 0;
            stats->loss+=one.loss;stats->index_loss+=one.index_loss;
            if(one.arena_bytes>stats->arena_bytes)stats->arena_bytes=one.arena_bytes;
        }
        return cn_optimizer_step(m,opt,stats);
    }
    /* Four complete sampled carry trajectories. All following column prompts
       use the sampled carry. The reward oracle is used only here in training. */
    if(!ref)return 0;
    enum { GROUP=4, MAXCOL=10, TOK=128 };
    int widths[]={2,4,10};int width=widths[(step-1)*3/(total?total:1)>2?2:(step-1)*3/(total?total:1)];
    int aa[MAXCOL],bb[MAXCOL],seq[GROUP][MAXCOL][TOK],len[GROUP][MAXCOL],pref[GROUP][MAXCOL],cols[GROUP]={0};
    for(int j=0;j<width;j++){aa[j]=(int)(add_random(rng)%10);bb[j]=(int)(add_random(rng)%10);}
    double rewards[GROUP]={0},sum=0;
    cn_context *actor=cn_context_create(m,0),*reference=cn_context_create(ref,0);
    if(!actor||!reference){cn_context_free(actor);cn_context_free(reference);return 0;}
    for(int k=0;k<GROUP;k++) {
        int carry=0,truth=0,all=1,correct=0;
        for(int j=0;j<width;j++) {
            int ids[5],valid=add_column_local(actor,aa[j],bb[j],carry,ids,rng,NULL);
            int *t=seq[k][j],p=0;
            if(!add_column_example(aa[j],bb[j],carry,0,t,TOK,&p))goto fail;
            int out=0;while(out<5&&ids[out]>=0){t[p+out]=ids[out];out++;}
            len[k][j]=p+out;pref[k][j]=p;cols[k]++;
            int expected=aa[j]+bb[j]+truth;
            int good=valid&&ids[0]-'0'==expected%10&&ids[2]-'0'==expected/10;
            correct+=good;all &= good;truth=expected/10;
            if(!valid){all=0;break;}
            carry=ids[2]-'0';
        }
        rewards[k]=.5*all+.5*correct/width;sum+=rewards[k];
    }
    *reward=sum/GROUP;
    double spread=0;for(int k=0;k<GROUP;k++)spread+=fabs(rewards[k]-sum/GROUP);
    if(spread<1e-9){cn_context_free(actor);cn_context_free(reference);return 1;}
    *informative=1;
    float logits[TOK*CN_VOCAB];int sequences=0;
    for(int k=0;k<GROUP;k++)for(int j=0;j<cols[k];j++) {
        int n=len[k][j],p=pref[k][j],*t=seq[k][j];
        cn_context_reset(reference);
        if(!cn_prefill(reference,t,(size_t)n-1,logits))goto fail;
        cn_token_objective obj[TOK]={{0}};
        float advantage=(float)(rewards[k]-(sum-rewards[k])/3);
        for(int q=p;q<n;q++) {
            obj[q].ce=advantage/GROUP;obj[q].kl=.001f/GROUP;obj[q].entropy=.00001f/GROUP;
            obj[q].reference_logits=logits+(q-1)*CN_VOCAB;
        }
        cn_metrics one;
        if(!cn_objective_backward(m,t,(size_t)n,obj,sequences++!=0,0,&one))goto fail;
        stats->loss+=one.loss;if(one.arena_bytes>stats->arena_bytes)stats->arena_bytes=one.arena_bytes;
    }
    cn_context_free(actor);cn_context_free(reference);return cn_optimizer_step(m,opt,stats);
fail:
    cn_context_free(actor);cn_context_free(reference);return 0;
}
static int natural(const char *s){if(!s||!*s)return 0;for(;*s;s++)if(*s<'0'||*s>'9')return 0;return 1;}
static void ids_json(FILE *f,const int *t,int n){fputc('[',f);for(int j=0;j<n;j++)fprintf(f,"%s%d",j?",":"",t[j]);fputc(']',f);}
static void render(FILE *f,const int *t,int n) {
    for(int j=0;j<n&&t[j]>=0;j++) {
        char text[64];size_t z=cn_decode(t+j,1,1,text,sizeof(text));
        if(z==SIZE_MAX||z>sizeof(text))continue;
        for(size_t k=0;k<z;k++){unsigned char c=(unsigned char)text[k];if(c=='\n'||(c>=32&&c<127))fputc(c,f);else fprintf(f,"\\x%02x",c);}
    }
}
int add_column_predict(cn_model *m,const char *a,const char *b,const char *jsonpath,const char *rawpath) {
    if(!m||!natural(a)||!natural(b))return 0;
    size_t na=strlen(a),nb=strlen(b),width=na>nb?na:nb;
    if(width>SIZE_MAX-3)return 0;
    char *digits=malloc(width+2),*carries=malloc(width+2);cn_context *s=cn_context_create(m,0);
    FILE *json=fopen(jsonpath,"wb"),*raw=fopen(rawpath,"wb");
    if(!digits||!carries||!s||!json||!raw){free(digits);free(carries);cn_context_free(s);if(json)fclose(json);if(raw)fclose(raw);return 0;}
    fputs("# structured-column-v2: actual network calls; position/digit extraction and answer assembly are scheduled.\n",raw);
    fprintf(raw,"# problem: %s+%s\n",a,b);
    fputs("{\"ok\":true,\"protocol\":\"column-v2\",\"mode\":\"cot\",\"columns\":[",json);
    int carry=0,valid=1;size_t written=0,generated=0;
    for(size_t pos=0;pos<width;pos++) {
        int da=pos<na?a[na-pos-1]-'0':0,db=pos<nb?b[nb-pos-1]-'0':0,ids[5];
        int good=add_column_local(s,da,db,carry,ids,NULL,NULL),t[128];
        int p=column_prompt(da,db,carry,0,t,128);
        fprintf(raw,"# pos=%zu a=%d b=%d cin=%d\n",pos,da,db,carry);render(raw,t,p);render(raw,ids,5);fputc('\n',raw);
        fprintf(json,"%s{\"pos\":%zu,\"a\":%d,\"b\":%d,\"cin\":%d,\"digit\":%d,\"cout\":%d,\"valid\":%s,\"input_ids\":",pos?",":"",pos,da,db,carry,good?ids[0]-'0':-1,good?ids[2]-'0':-1,good?"true":"false");
        ids_json(json,t,p);fputs(",\"ids\":",json);ids_json(json,ids,5);fputc('}',json);
        for(int j=0;j<5;j++)generated+=ids[j]>=0;
        if(!good){valid=0;break;}
        carries[written]=(char)(ids[2]-'0');digits[written++]=(char)ids[0];carry=ids[2]-'0';
    }
    /* This presentation is explicitly scheduled; actual per-call messages above
       are not replaced by an invented monolithic model input/output. */
    fputs("# scheduled whole-problem CoT assembled from model predictions:\n<think>\n",raw);
    for(size_t pos=0;pos<written;pos++)fprintf(raw,"pos=%zu a=%d b=%d cin=%d -> %c %d\n",pos,
        pos<na?a[na-pos-1]-'0':0,pos<nb?b[nb-pos-1]-'0':0,pos?(int)carries[pos-1]:0,digits[pos],(int)carries[pos]);
    fputs("</think>\n",raw);
    if(valid&&carry)digits[written++]=(char)('0'+carry);
    while(valid&&written>1&&digits[written-1]=='0')written--;
    fputs("],\"valid\":",json);fputs(valid?"true":"false",json);fputs(",\"answer\":\"",json);
    if(valid)for(size_t j=written;j>0;j--)fputc(digits[j-1],json);
    fprintf(json,"\",\"error\":\"%s\",\"generated\":%zu,\"rollovers\":0}",valid?"":"column_format",generated);
    int ok=!ferror(json)&&!ferror(raw);if(fclose(json))ok=0;if(fclose(raw))ok=0;
    cn_context_free(s);free(digits);free(carries);return ok;
}
