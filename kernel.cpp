#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################

/**
 * Kernel function for the cfd
 */
kernel void tick_lattice(global double* Lattice,global double* LatticeOutput,global double* Size,global int* boundary
                         ,global double* weight,global double* dAir,global int* offsetl,global double* tMatrix,global double* Record) {
	const uint n = get_global_id(0);
	double omega =  0.809;
	double omega_bulk =  0.809;
	double diagomega=1./Size[3];
    double omegamat[]={2.0,omega_bulk,omega,0.0,omega,0.0,omega,omega,omega};


	uint Nx = Size[0];
	uint Ny = Size[1];
	uint Nl = Size[2];

	if(n > Nx * Ny) return;
	uint x = n%Nx; //this thread's x
	uint y = n/Nx; //this thread's y

    if(x == 0 || y == 0 || x == Nx - 1 || y == Ny - 1){
        for(int i = 0;i < Nl;i++){
            uint ci = n*Nl+i;
            LatticeOutput[ci] = dAir[i];
        }
        return;
    }

    if(boundary[n]==2){
        for(int i = 0;i < Nl;i++){
            uint ci = n*Nl+i;
            LatticeOutput[ci] = 0;
        }
        return;
    }
	//Streaming
	for(int i = 0;i < Nl;i++){//loop around neighboring elements and apply to self
        uint ci = n*Nl+i;
	    int offsetx = offsetl[i];
	    int offsety = offsetl[i+Nl];
        if((int)x-offsetx<0||(int)x-offsetx>=Nx||(int)y-offsety<0||(int)y-offsety>=Ny) continue;
        LatticeOutput[ci] = Lattice[ci-offsetx*Nl-offsety*Nl*Nx];
	}




    //Fluid Variables
    double density = 0;
    double velocityx = 0;
    double velocityy = 0;
    for(int i = 0;i < Nl;i++){
        int offsetx = offsetl[i];
	    int offsety = offsetl[i+Nl];
        density += LatticeOutput[n*Nl+i];
        velocityx += offsetx * LatticeOutput[n*Nl+i];//momentum
        velocityy += offsety * LatticeOutput[n*Nl+i];
    }
    /*if(density <= 0 || density >= 500000){
        velocityx = 0;
        velocityy = 0;
        double densityoffset = -density;
        for(int i = 0;i < Nl;i++){
            int offsetx = offsetl[i];
            int offsety = offsetl[i+Nl];
            LatticeOutput[n*Nl+i] += densityoffset/9;
            velocityx += offsetx * LatticeOutput[n*Nl+i];//momentum
            velocityy += offsety * LatticeOutput[n*Nl+i];
        }
    }*/

    velocityx /= density;//velocity
    velocityy /= density;


    //Boundary Reflection
    //for(int i = 0;i < Nx*Ny;i++){
        //printf("%d\n",n);
    //}
    if(boundary[n]==1){
        bool l[9] = {0};
        for(int i = 0;i < Nl;i++){
            if(l[i] == true) continue;
            uint ci1 = n*Nl+i;
            uint ci2 = n*Nl+offsetl[i+Nl*3];
            l[offsetl[i+Nl*3]] = true;
            double t = LatticeOutput[ci1];
            LatticeOutput[ci1] = LatticeOutput[ci2];
            LatticeOutput[ci2] = t;
        }
        //LatticeOutput[n*Nl+4] = 10;
        //LatticeOutput[n*Nl+5] = n;
    }

	//Collision

	else{
/*
        for(int i = 0;i < Nl;i++){

            uint ci = n*Nl+i;
            int offsetx = offsetl[i];
            int offsety = offsetl[i+Nl];
            double equalizingF = density*weight[i]*(1. + 3.*(double)(offsetx*velocityx+offsety*velocityy)
                    + 9. *(double)(offsetx*velocityx+offsety*velocityy)*(offsetx*velocityx+offsety*velocityy) / 2.
                    - 3. * (double)(velocityx*velocityx+velocityy*velocityy)/2.);


            LatticeOutput[ci] += (1./Size[3])*(equalizingF-LatticeOutput[ci]);
            //LatticeOutput[ci] = equalizingF;
        }
        /**/
        // velocity space calc, unstable

        double* M = tMatrix;
        double* invM = tMatrix+Nl*Nl;
        //Momentspace equilib
        //Construction of the equilibrium moments
        double eq[9];
        eq[0]=density;
        eq[1]=-2.0*density+3.0*density*(velocityx*velocityx+velocityy*velocityy);
        eq[2]=density-3.0*density*(velocityx*velocityx+velocityy*velocityy);
        eq[3]=density*velocityx;
        eq[4]=-density*velocityx;
        eq[5]=density*velocityy;
        eq[6]=-density*velocityy;
        eq[7]=density*(velocityx*velocityx-velocityy*velocityy);
        eq[8]=density*velocityx*velocityy;
        //eq[0]=density;
        //eq[1]=sqrt(3.0)*density*velocityx;
        //eq[2]=sqrt(3.0)*density*velocityy;
        //eq[3]=3.0/sqrt(2.0)*density*velocityx*velocityx;
        //eq[4]=3.0*density*velocityx*velocityy;
        //eq[5]=3.0/sqrt(2.0)*density*velocityy*velocityy;
        //eq[6]=4.5*density*velocityx*velocityx*velocityy*velocityy;
        //eq[7]=3.0*sqrt(1.5)*density*velocityx*velocityy*velocityy;
        //eq[8]=3.0*sqrt(1.5)*density*velocityx*velocityx*velocityy;

        double add[9];
        double addit;
        for(int iPop=0;iPop < 9; iPop++)
        {
            add[iPop]=0.0;
            for(int k=0; k < 9; k++)
                add[iPop]+=M[iPop+k*Nl]*(-LatticeOutput[n*Nl+k]);
            add[iPop]+=eq[iPop];
            add[iPop]*=omegamat[iPop];
        }

        //Momentspace to vectorspace

        for(int i = 0;i < Nl;i++){

            uint ci = n*Nl+i;
            int offsetx = offsetl[i];
            int offsety = offsetl[i+Nl];

            addit=0.0;
            for(int m=0; m < 9; m++)
                addit=addit+invM[i+m*Nl]*add[m];
            LatticeOutput[ci] += weight[i] *  addit;
            //LatticeOutput[ci] = equalizingF;
        }/**/
	}

	//LatticeOutput[n*Nl+4] = boundary[n];





}
kernel void copyarr(global double* Lattice,global double* LatticeEnd,global double* Size){
    const uint n = get_global_id(0);
    uint Nx = Size[0];
	uint Ny = Size[1];
	uint Nl = Size[2];
	if(n > Nx*Ny*Nl) return;
    Lattice[n] = LatticeEnd[n];
}



);} // ############################################################### end of OpenCL C code #####################################################################
