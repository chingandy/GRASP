

std::vector<approx_circel> c_line(int x, int y, double t,int r=2)
{
	//build a line out of circles
	//rotation point is entrie point
	std::vector<approx_circel> c_line;
	approx_circel c_temp;
	c_temp.r=r;
	double dh=2.0;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_line.push_back(c_temp);
	}
	return c_line;

}

std::vector<approx_circel> c_l_shape(int x, int y, double t,int r=2)
{
	//build a L-shape out of circles
	std::vector<approx_circel> c_l_shape;
	approx_circel c_temp;

	c_temp.r=r;
	double dh=2.0;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_l_shape.push_back(c_temp);
	}
	t=t+M_PI/2;
	std::cout << t << std::endl;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_l_shape.push_back(c_temp);
	}


	return c_l_shape;
}

std::vector<approx_circel> c_u_shape(int x, int y, double t,int r=2)
{
	//build a U-shape out of circles
	std::vector<approx_circel> c_u_shape;
	approx_circel c_temp;
	c_temp.r=r;
	
	double dh=2.0;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_u_shape.push_back(c_temp);
	}
	t=t+M_PI/2;
	std::cout << t << std::endl;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_u_shape.push_back(c_temp);
	}
	x=c_u_shape[c_u_shape.size()-1].x;
	y=c_u_shape[c_u_shape.size()-1].y;
	t=t-M_PI/2;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_u_shape.push_back(c_temp);
	}


	return c_u_shape;

}

std::vector<approx_circel> c_sqr(int x, int y, double t,int r=2)
{
	//build a square out of circles
	std::vector<approx_circel> c_sqr;
	approx_circel c_temp;
	c_temp.r=r;
	
	double dh=2.0;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_sqr.push_back(c_temp);
	}
	t=t+M_PI/2;
	std::cout << t << std::endl;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_sqr.push_back(c_temp);
	}
	x=c_sqr[c_sqr.size()-1].x;
	y=c_sqr[c_sqr.size()-1].y;
	t=t-M_PI/2;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_sqr.push_back(c_temp);
	}
	x=c_sqr[c_sqr.size()-1].x;
	y=c_sqr[c_sqr.size()-1].y;
	t=t-M_PI/2;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_sqr.push_back(c_temp);
	}

	//fill out the middelpart ...
	t=t-M_PI/4;
	c_temp.x=rint(cos(t)*(dh*4)+x);
	c_temp.y=rint(sin(t)*(dh*4)+y);
	c_temp.r=r*4;
	c_sqr.push_back(c_temp);

	return c_sqr;

}

std::vector<approx_circel> c_c_shape(int x, int y, double t,int r=2)
{
	//build a C-shape out of circles
	std::vector<approx_circel> c_c_shape;
	approx_circel c_temp;
	c_temp.r=r;

	
	double dh=10.0;
	//calculate new midelpoint
	x=x-rint(cos(t)*(dh));
	y=y-rint(sin(t)*(dh));
	for(int i=0;i<10;i++)
	{
		c_temp.x=rint(cos(t)*(dh)+x);
		c_temp.y=rint(sin(t)*(dh)+y);
		t=t+M_PI/10;
		c_c_shape.push_back(c_temp);
	}

	return c_c_shape;

}

std::vector<approx_circel> c_hc_shape(int x, int y, double t,int r=2)
{
	//build a half-C-shape out of circles
	std::vector<approx_circel> c_hc_shape;
	approx_circel c_temp;
	c_temp.r=r;
	
	double dh=10.0;
	//calculate new midelpoint
	x=x-rint(cos(t)*(dh));
	y=y-rint(sin(t)*(dh));
	for(int i=0;i<5;i++)
	{
		c_temp.x=rint(cos(t)*(dh)+x);
		c_temp.y=rint(sin(t)*(dh)+y);
		t=t+M_PI/(2*5);
		c_hc_shape.push_back(c_temp);
	}

	return c_hc_shape;

}

std::vector<approx_circel> c_triangle(int x, int y, double t,int r=2)
{
	//build a triangle out of circles
	std::vector<approx_circel> c_triangle;
	approx_circel c_temp;
	c_temp.r=r;
	
	double dh=2.0;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_triangle.push_back(c_temp);
	}
	t=t+M_PI/5;
	std::cout << t << std::endl;
	for(int i=0;i<8;i++)
	{
		c_temp.x=rint(cos(t)*(dh*i)+x);
		c_temp.y=rint(sin(t)*(dh*i)+y);
		c_triangle.push_back(c_temp);
	}
	//final blob
	t=t-(M_PI/5)/2;
	c_temp.x=rint(cos(t)*(dh*6)+x);
	c_temp.y=rint(sin(t)*(dh*6)+y);
	c_temp.r=r*2;
	c_triangle.push_back(c_temp);

	return c_triangle;


}

std::vector<approx_circel> c_disk(int x, int y, int r)
{
	//simple disk
	std::vector<approx_circel> c_disk;
	approx_circel c_temp;
	c_temp.x=x;
	c_temp.y=y;
	c_temp.r=r;	
	c_disk.push_back(c_temp);

	return c_disk;
}
