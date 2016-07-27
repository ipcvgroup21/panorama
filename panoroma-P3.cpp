
#include <stdio.h>
#include <math.h>

typedef struct key_points
{
public:
	double x;
	double y;
	double sigma;
	int oct_id;
	int scale_id;
	double orientation;
	std::vector<double> *v;
} key_point;


void GetOriAndMag(int oct_id, int scale_id)
{
	double PI=3.14159265358;
	Mat origin=Pyramid[oct_id][scale_id];
	int w = origin.width(), h = origin.height();
	mag_Pyramid[oct_id][scale_id]= Mat(h,w,CV_64FC1,0);
	ori_Pyramid[oct_id][scale_id]= Mat(h,w,CV_64FC1,0);
	for(y=0;y<h;y++)
	{
		double *mag_row = mag_Pyramid[oct_id][scale_id].pty(y);
		double *ori_row = ori_Pyramid[oct_id][scale_id].pty(y);
		double *origin_row = origin.pty(y), *origin_uprow = origin.pty(y+1), *origin_downrow = origin.pty(y-1);

		mag_row[0]=0;
		ori_row[0]=PI;

		for(x=1;x<w-1;x++)
		{
			if(y>0 && y<h-1)
			{
				double dy=origin_uprow[x]-origin_downrow[x];
				double dx=ori_row[x+1]-ori_row[x-1];
				mag_row[x]=sqrt(dx*dx+dy*dy);
				ori_row[x]=atan2(dy,dx);
				if(ori_row[x]<0) ori_row[x] += 2*PI;
			}
			else
			{
				mag_row[x]=0;
				ori_row[x]=PI;
			}
		}

		mag_row[w-1]=0;
		ori_row[w-1]=PI;
	}
}


std::vector<double> OrientationAssignment(key_point p)
{
	Mat ori_img=ori_Pyramid[p.oct_id][p.scale_id];
	Mat mag_img=mag_Pyramid[p.oct_id][p.scale_id];
	double PI=3.14159265358;
	int r=round(p.sigma*4.5);
	double hist[36];
	for(int i=0;i<36;i++) hist[i]=0;

	for(int xx=-r;xx<=r;xx++)
	{
		int x=p.x+xx;
		if(x<=0 || x>=ori_img.width()-1) continue;
		for(int yy=-r;yy<=r;yy++)
		{
			int y=p.y+yy;
			if(y<=0 || y>=ori_img.height()-1) continue;
			if(sqr(xx)+sqr(yy)>sqr(r)) continue;
			double ori=ori_img.at(y,x);
			int bin=round(ori/PI*180.0/10.0);
			if (bin==36) bin=0;
			hist[bin]+=mag_img.at(y,x)*exp( -(sqr(xx)+sqr(yy)) / (2*sqr(1.5*p.sigma)) );

		}
	}

	for(i=0;i<2;i++)
	{
		double hist_tmp[36];
		for(j=0;j<36;j++)
		{
			double prev=hist[j==0? 35:j-1];
			double next=hist[j==35? 0:j+1];
			hist_tmp[j]=hist[j]*0.5+(prev+next)*0.25;
		}
		for(j=0;j<36;j++)
		{
			hist[j]=hist_tmp[j];
		}
	}

	double hist_max=0;
	for(int i=0;i<36;i++)
	{
		if(hist[i]>hist_max) hist_max=hist[i];
	}
	double threshold=0.8*hist_max;
	std::vector<double> result;

	for(int i=0;i<36;i++)
	{
		double prev=hist[j==0? 35:j-1];
		double next=hist[j==35? 0:j+1];

		if(hist[i]>threshold && hist[i]>prev && hist[i]>next)
		{
			double a=(next+prev-2*hist[i])*0.5;
			double b=(next-prev)*0.5;
			
			double bin_real = i*1.0 - 0.5*b/a;

			if(bin_real<0) bin_real+=36;

			double orientation = bin_real * 10.0 / 180.0 * PI;
			result.push_back(orientation);
		}
	}
	return result;
}


std::vector<double> GetVector(key_point p)
{
	double PI=3.14159265358;
	Mat mag_img=mag_Pyramid[p.oct_id][p.scale_id];
	Mat ori_img=ori_Pyramid[p.oct_id][p.scale_id];
	int w=ori_img.width(), h=ori_img.height();

	double ori=p.orientation;
	int r=round(sqrt(0.5)*3*p.sigma*(4+1));
	double hist[16][8];
	for(int i=0;i<16;i++)
		for(int j=0;j<8;j++)
			hist[i][j]=0;

	double cos_part= cos(ori), sin_part= sin(ori);

	for(int xx=-r;xx<=r;xx++)
	{
		int x=p.x+xx;
		if(x<=0 || x>=w-1) continue;
		for(int yy=-r;yy<=r;yy++)
		{
			int y=p.y+yy;
			if(y<=0 || y>=h-1) continue;
			if(sqr(xx)+sqr(yy)>sqr(r)) continue;

			double bin_y = (-xx*sin_part+yy*cos_part)/(3.0*p.sigma)+1.5;
			double bin_x = (xx*cos_part +yy*sin_part)/(3.0*p.sigma)+1.5;

			if(bin_y<-1 || bin_y>=4 || bin_x<-1 || bin_x>=4) continue;

			double ori_loc = ori_img.at(y,x);
			double mag_loc = mag_img.at(y,x);

			double w_mag = mag_loc * exp( -(sqr(xx)+sqr(yy)) / 8 );
			double r_ori -= ori;

			if(r_ori<0) r_ori += 2*PI;

			double bin_hist = r_ori /PI * 180.0 /45.0;

			TriInterpolation(bin_x,bin_y,bin_hist,w_mag,hist);
 		}
	}
	std::vector<double> result(128);
	for(int i=0;i<16;i++)
		for(int j=0;j<8;j++)
			result[8*i+j]=hist[i][j];

	double sum=0;
	for(int i=0;i<128;i++) sum+=sqr(result[i]);
	sum=sqrt(sum);
    for(int i=0;i<128;i++)
    {
    	result[i]/=sum;
    	if(result[i]>0.2) result[i]=0.2;
    }
    sum=0;
    for(int i=0;i<128;i++) sum+=sqr(result[i]);
	sum=sqrt(sum);
    for(int i=0;i<128;i++) result[i]/=sum;

    return result;
}

void TriInterpolation(double x, double y, double h, double w_mag, double hist[][8])
{
	int xf=floor(x),
	    yf=floor(y),
	    hf=floor(h);

	double delta_x = x-xf,
	       delta_y = y-yf,
	       delta_h = h-hf;

	for(int i=0;i<2;i++)
	{
		if((yf+i)>=0 && (yf+i)<4)
		{
			double wy = w_mag * (i? delta_y : 1-delta_y);
			for(j=0;j<2;j++)
			{
				if((xf+j)>=0 && (xf+j)<4)
				{
					double wx = wy * (j? delta_x : 1-delta_x);
					hist[4*(yf+i)+(xf+j)][hf % 8] += wx * (1-delta_h);
					hist[4*(yf+i)+(xf+j)][(hf+1) % 8] += wx * delta_h;
				}
			}
		}
	}
}