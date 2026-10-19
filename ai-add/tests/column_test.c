#include "add_column.h"
#include "add_session.h"
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"column FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(int argc,char **argv){
    CHECK(argc==2);int t[128],p=0;char text[1024]={0};
    int n=add_column_example(7,1,0,0,t,128,&p);CHECK(n>p);
    CHECK(cn_decode(t,n,1,text,sizeof(text)-1)<sizeof(text));
    CHECK(!strcmp(text,"<|im_start|>user\nadd a=7 b=1 cin=0<|im_end|>\n<|im_start|>assistant\n8 0<|im_end|><|endoftext|>"));
    for(int i=0;i<n;i++)CHECK(t[i]!=CN_THINK&&t[i]!=CN_THINK_END);
    cn_config c;add_chat_config(&c,20261018);
    add_session*s=add_session_create(&c,"column-smoke");CHECK(s&&add_session_set_chat(s,2));
    for(int i=1;i<=6;i++){CHECK(add_session_begin(s,i,1,1,0));CHECK(add_session_tick(s)==1);CHECK(add_session_progress(s)->correct>=add_session_progress(s)->last_correct);}
    CHECK(add_column_predict(add_session_model(s),"9","9","column-failed.json","column-failed.txt"));
    add_session_free(s);char error[256];cn_model*m=cn_load(argv[1],error,sizeof(error));CHECK(m);
    add_chat_validation r;CHECK(add_column_validate(m,&r)&&r.exact.correct==200&&r.exact.margin>0);
    CHECK(add_column_predict(m,"9999999999","1","column-good.json","column-good.txt"));
    FILE*f=fopen("column-good.json","rb");CHECK(f);size_t got=fread(text,1,sizeof(text)-1,f);text[got]=0;fclose(f);CHECK(strstr(text,"\"protocol\":\"column-v2\""));
    cn_model_free(m);puts("real role messages, complete local endings, six stages, failed output and trained transitions passed");return 0;
}
