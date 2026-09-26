
#include <cmath>
#include <algorithm>
#include <vector>
struct V {double x,y,z;};
static bool lineclear(const float*dem,int nr,int nc,V a,V b){
 double dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z;
 if(fabs(dx)+fabs(dy)<1e-11){int x=floor(a.x),y=floor(a.y);return x>=0&&x<nc&&y>=0&&y<nr&&dem[y*nc+x]<std::min(a.z,b.z)-1e-7;}
 std::vector<double> ts{0.,1.};
 if(fabs(dx)>1e-12)for(int k=ceil(std::min(a.x,b.x));k<=floor(std::max(a.x,b.x));k++){double t=(k-a.x)/dx;if(t>0&&t<1)ts.push_back(t);}
 if(fabs(dy)>1e-12)for(int k=ceil(std::min(a.y,b.y));k<=floor(std::max(a.y,b.y));k++){double t=(k-a.y)/dy;if(t>0&&t<1)ts.push_back(t);}
 std::sort(ts.begin(),ts.end());
 for(int j=0;j+1<(int)ts.size();j++) {double t=(ts[j]+ts[j+1])/2;int x=floor(a.x+t*dx),y=floor(a.y+t*dy);if(x<0||x>=nc||y<0||y>=nr)return false;double low=std::min(a.z+ts[j]*dz,a.z+ts[j+1]*dz);if(dem[y*nc+x]>=low-1e-7)return false;}
 // Cell-boundary contacts: include every adjacent closed pixel.
 for(double t:ts){double xx=a.x+t*dx,yy=a.y+t*dy,zz=a.z+t*dz;int x=floor(xx),y=floor(yy);for(int h=-1;h<=1;h++)for(int k=-1;k<=1;k++){int c=x+k,r=y+h;if(c<0||c>=nc||r<0||r>=nr)continue;if(xx>=c-1e-9&&xx<=c+1+1e-9&&yy>=r-1e-9&&yy<=r+1+1e-9&&dem[r*nc+c]>=zz-1e-7)return false;}}
 return true;
}
static std::vector<V> clip(std::vector<V> p,int axis,double b,bool lower){
 std::vector<V>o;if(p.empty())return o;V prev=p.back();auto val=[&](V v){return axis?v.y:v.x;};
 bool pin=lower?val(prev)>=b-1e-10:val(prev)<=b+1e-10;
 for(V cur:p){bool in=lower?val(cur)>=b-1e-10:val(cur)<=b+1e-10;if(in!=pin){double den=val(cur)-val(prev);if(fabs(den)>1e-16){double t=(b-val(prev))/den;o.push_back({prev.x+t*(cur.x-prev.x),prev.y+t*(cur.y-prev.y),prev.z+t*(cur.z-prev.z)});}}if(in)o.push_back(cur);prev=cur;pin=in;}return o;
}
static bool triclear(const float*dem,int nr,int nc,V a,V b,V c){
 double det=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
 if(fabs(det)<1e-9)return lineclear(dem,nr,nc,a,b)&&lineclear(dem,nr,nc,a,c)&&lineclear(dem,nr,nc,b,c);
 std::vector<V>tri{a,b,c};double zmin=std::min({a.z,b.z,c.z});
 int ya=floor(std::min({a.y,b.y,c.y})-1e-9),yb=floor(std::max({a.y,b.y,c.y})+1e-9);
 for(int y=ya;y<=yb;y++){
  auto strip=clip(clip(tri,1,y,true),1,y+1,false);if(strip.empty())continue;
  double xmin=1e30,xmax=-1e30;for(V v:strip){xmin=std::min(xmin,v.x);xmax=std::max(xmax,v.x);}
  for(int x=floor(xmin-1e-9);x<=floor(xmax+1e-9);x++){
   if(x<0||x>=nc||y<0||y>=nr)return false;
   double z=dem[y*nc+x];if(!std::isfinite(z)||z<=-9999)return false;if(z<zmin-1e-7)continue;
   auto p=clip(clip(strip,0,x,true),0,x+1,false);if(p.empty())continue;
   double low=1e30;for(V v:p)low=std::min(low,v.z);if(z>=low-1e-7)return false;
  }
 }
 return true;
}
extern "C" {
 int rayclear(const float*dem,int nr,int nc,const double*a,const double*b){return lineclear(dem,nr,nc,{a[0],a[1],a[4]},{b[0],b[1],b[4]});}
 int sweptclear(const float*dem,int nr,int nc,const double*a,const double*b,const double*c){return triclear(dem,nr,nc,{a[0],a[1],a[4]},{b[0],b[1],b[4]},{c[0],c[1],c[4]});}
 void pointcover(const float*dem,int nr,int nc,const double*anchor,const double*points,int n,double limit,unsigned char*out){
 for(int i=0;i<n;i++){const double*p=points+5*i;double d=hypot(hypot(anchor[2]-p[2],anchor[3]-p[3]),anchor[4]-p[4]);double fs=32.45+20*log10(2400.)+20*log10(std::max(d,1e-5)/1000.);if(fs+10<=limit-1e-9)out[i]=1;else if(fs>limit-1e-9)out[i]=0;else out[i]=lineclear(dem,nr,nc,{anchor[0],anchor[1],anchor[4]},{p[0],p[1],p[4]});}}
}
