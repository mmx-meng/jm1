
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
using P=std::array<double,2>;
static std::vector<P> clip(const std::vector<P>& p,double a,double b,double c){
 std::vector<P>o;if(p.empty())return o;P prev=p.back();double vp=a*prev[0]+b*prev[1]+c;bool pin=vp>=0;
 for(P cur:p){double vc=a*cur[0]+b*cur[1]+c;bool in=vc>=0;if(in!=pin){double den=vp-vc;if(fabs(den)>1e-30){double w=vp/den;o.push_back({prev[0]+w*(cur[0]-prev[0]),prev[1]+w*(cur[1]-prev[1])});}}if(in)o.push_back(cur);prev=cur;vp=vc;pin=in;}return o;
}
extern "C" int blockevents(const float*dem,int nr,int nc,const double*A,const double*B,const double*C,double*out,int capacity){
 double ax=A[0],ay=A[1],az=A[4],bx=B[0]-ax,by=B[1]-ay,bz=B[4]-az,dx=C[0]-B[0],dy=C[1]-B[1],dz=C[4]-B[4];
 double minz=std::min({az,B[4],C[4]});std::vector<P>times;
 // Raster rows are pruned with the projected swept triangle (possibly degenerate).
 std::vector<P>xy{{A[0],A[1]},{B[0],B[1]},{C[0],C[1]}};
 int ya=floor(std::min({A[1],B[1],C[1]})-1e-9),yb=floor(std::max({A[1],B[1],C[1]})+1e-9);
 for(int y=ya;y<=yb;y++){
  auto strip=clip(clip(xy,0,1,-y+1e-10),0,-1,y+1+1e-10);if(strip.empty())continue;
  double xmin=1e30,xmax=-1e30;for(P p:strip){xmin=std::min(xmin,p[0]);xmax=std::max(xmax,p[0]);}
  for(int x=floor(xmin-1e-9);x<=floor(xmax+1e-9);x++){
   if(x<0||x>=nc||y<0||y>=nr)return -2;
   double h=dem[y*nc+x];if(!std::isfinite(h)||h<=-9999)return -3;if(h+1e-7<minz)continue;
   if(ax>=x-1e-10&&ax<=x+1+1e-10&&ay>=y-1e-10&&ay<=y+1+1e-10&&az<=h+1e-7){out[0]=0;out[1]=1;return 1;}
   std::vector<P>p{{0,0},{1,0},{1,1}};
   p=clip(p,bx,dx,ax-x+1e-10);p=clip(p,-bx,-dx,x+1-ax+1e-10);
   p=clip(p,by,dy,ay-y+1e-10);p=clip(p,-by,-dy,y+1-ay+1e-10);
   p=clip(p,-bz,-dz,h+1e-7-az);if(p.empty())continue;
   double lo=1e30,hi=-1e30;
   for(P v:p)if(v[0]>1e-14){double t=v[1]/v[0];lo=std::min(lo,t);hi=std::max(hi,t);}
   if(hi>=lo&&hi>=-1e-12&&lo<=1+1e-12)times.push_back({std::max(0.,lo),std::min(1.,hi)});
  }
 }
 std::sort(times.begin(),times.end(),[](P a,P b){return a[0]<b[0]||(a[0]==b[0]&&a[1]<b[1]);});
 std::vector<P>u;for(P t:times){if(!u.empty()&&t[0]<=u.back()[1]+1e-13)u.back()[1]=std::max(u.back()[1],t[1]);else u.push_back(t);}
 if((int)u.size()>capacity)return -(10000+(int)u.size());for(int i=0;i<(int)u.size();i++){out[2*i]=u[i][0];out[2*i+1]=u[i][1];}return u.size();
}
