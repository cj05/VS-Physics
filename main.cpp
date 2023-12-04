#include "opencl.hpp"
#include "iostream"
#include "cmath"
#define printArr(n,N) for(int i = 0;i < N/Nx/3;i++) {for(int j = 0;j < Nx*3;j++) std::cout<<n[(i%3*3+j%3) + (j/3*9+i/3*Nx*9)]<<" "; std::cout<<"\n";} std::cout<<"\n"; std::cout<<std::flush;

#define printCFDDG(n) for(int j = 0;j < Ny;j++) {for(int i = 0;i < Nx;i++){ double d = 0; for(int k = 0;k < Nl;k++) d += n[(j*Nx+i)*Nl+k]; std::cout<<gradient[69-std::min(69,int(d*20))]<<" ";} std::cout<<"\n";} std::cout<<"\n"; std::cout<<std::flush;

#define printCFDD(n) for(int j = 0;j < Ny;j++) {for(int i = 0;i < Nx;i++){ double d = 0; for(int k = 0;k < Nl;k++) d += n[(j*Nx+i)*Nl+k]; std::cout<<d<<" ";} std::cout<<"\n";} std::cout<<"\n"; std::cout<<std::flush;
//


char gradient[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
class BoltzmanLattice{
    private:
        const uint Nx = 400;
        const uint Ny = 100;
        const uint Nl = 9;
        double avgdensity  = 1.225;    // average density
        double tau         = 0.5;    // collision timescale
        Device device;
        Memory<double> *Lattice;
        Memory<double> *LatticeEnd;
        Memory<int> *boundary;
        Memory<double> *weight;
        Memory<double> *Size;
        Memory<double> *dAir;
        Memory<int> *cOffset;
        Memory<double> *tMatrix;
        int cx[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
        int cy[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1};
        double* w;

        void init(Device device,int* boundaryIn,double* enviromentAir){




            double density = 0;
            for(int i = 0; i < Nl;i++){
                density += enviromentAir[i];
            }

            this->Lattice = new Memory<double>(device, this->Nx*this->Ny*this->Nl); // 3darray in 1d, access data at l+(y+x*Ny)*Nl
            this->LatticeEnd = new Memory<double>(device, this->Nx*this->Ny*this->Nl); // 3darray in 1d, access data at l+(y+x*Ny)*Nl
            this->boundary = new Memory<int>(device, this->Nx*this->Ny);
            this->weight = new Memory<double>(device, Nl);
            this->Size = new Memory<double>(device,4);
            this->dAir = new Memory<double>(device,Nl,1,enviromentAir);
            this->cOffset = new Memory<int>(device,Nl*4);
            this->tMatrix = new Memory<double>(device,Nl*Nl*2);

            double M[(this->Nl)*(this->Nl)] = {};
            double invM[(this->Nl)*(this->Nl)] = {};




            const double w[] = {4/9.,1/9.,1/9.,1/9.,1/9.,1/36.,1/36.,1/36.,1/36.};

            //int cz[] = { 1, 1, 1, 0, 0, 0,-1,-1,-1};
            int inverse[] = { 0, 3, 4, 1, 2, 7, 8, 5, 6};
            double weightsmat[]={1.0/9.0,1.0/36.0,1.0/36.0,1.0/6.0,1.0/12.0,1.0/6.0,1.0/12.0,1.0/4.0,1.0/4.0};
            double g[]={1.0,-2.0,-2.0,-2.0,-2.0,4.0,4.0,4.0,4.0};
            for(int i = 0; i < (this->Nl);i ++){
                M[0+i*Nl]=1.0;
                M[1+i*Nl]=-4.0+3.0*(cx[i]*cx[i]+cy[i]*cy[i]);
                M[2+i*Nl]=4.0-21.0/2.0*(cx[i]*cx[i]+cy[i]*cy[i])+9.0/2.0*(cx[i]*cx[i]+cy[i]*cy[i])*(cx[i]*cx[i]+cy[i]*cy[i]);
                M[3+i*Nl]=cx[i];
                M[4+i*Nl]=(-5.0+3.0*(cx[i]*cx[i]+cy[i]*cy[i]))*cx[i];
                M[5+i*Nl]=cy[i];
                M[6+i*Nl]=(-5.0+3.0*(cx[i]*cx[i]+cy[i]*cy[i]))*cy[i];
                M[7+i*Nl]=cx[i]*cx[i]-cy[i]*cy[i];
                M[8+i*Nl]=cx[i]*cy[i];
                /*M[0+i*Nl]=1.0;
                M[1+i*Nl]=cx[i]*sqrt(3.0);
                M[2+i*Nl]=cy[i]*sqrt(3.0);
                M[3+i*Nl]=(cx[i]*cx[i]-1.0/3.0)*3.0/sqrt(2.0);
                M[4+i*Nl]=cx[i]*cy[i]*3.0;
                M[5+i*Nl]=(cy[i]*cy[i]-1.0/3.0)*3.0/sqrt(2.0);
                M[6+i*Nl]=g[i]/2.0;
                M[7+i*Nl]=g[i]*cx[i]*sqrt(1.5)/2.0;
                M[8+i*Nl]=g[i]*cy[i]*sqrt(1.5)/2.0;*/

                invM[i+0*Nl]=weightsmat[0]*1.0;
                invM[i+1*Nl]=weightsmat[1]*(-4.0+3.0*(cx[i]*cx[i]+cy[i]*cy[i]));
                invM[i+2*Nl]=weightsmat[2]*(4.0-21.0/2.0*(cx[i]*cx[i]+cy[i]*cy[i])+9.0/2.0*(cx[i]*cx[i]+cy[i]*cy[i])*(cx[i]*cx[i]+cy[i]*cy[i]));
                invM[i+3*Nl]=weightsmat[3]*cx[i];
                invM[i+4*Nl]=weightsmat[4]*(-5.0+3.0*(cx[i]*cx[i]+cy[i]*cy[i]))*cx[i];
                invM[i+5*Nl]=weightsmat[5]*cy[i];
                invM[i+6*Nl]=weightsmat[6]*(-5.0+3.0*(cx[i]*cx[i]+cy[i]*cy[i]))*cy[i];
                invM[i+7*Nl]=weightsmat[7]*(cx[i]*cx[i]-cy[i]*cy[i]);
                invM[i+8*Nl]=weightsmat[8]*cx[i]*cy[i];
            }
            for(int i = 0; i < (this->Nl);i ++){
                for(int j = 0; j < (this->Nl);j ++){
                    std::cout<<M[j*Nl+i]<<" ";
                }
                std::cout<<"\n";
            }

            for(int i = 0; i < (this->Nl);i++){
                (*this->weight)[i] = w[i];
                (*this->cOffset)[i] = cx[i];
                (*this->cOffset)[i+Nl] = cy[i];
                (*this->cOffset)[i+Nl*3] = inverse[i];

            }

            for(int j = 0; j < (this->Nl)*(this->Nl);j++){
                (*this->tMatrix)[j] = M[j];
                (*this->tMatrix)[j+(this->Nl)*(this->Nl)] = invM[j];
            }

            (*this->Size)[0] = this->Nx;
            (*this->Size)[1] = this->Ny;
            (*this->Size)[2] = this->Nl;
            (*this->Size)[3] = this->tau;
            for(int i = 0; i < (this->Nx)*(this->Ny)*(this->Nl);i++){
                int l = i%this->Nl;
                (*this->Lattice)[i] =(*this->dAir)[l];
                //Lattice[i] = i;
                (*this->LatticeEnd)[i] = 0;
            }

            //this->boundarygen(boundaryIn);
            for(int i = 0;i < this->Ny; i++){
                for(int j = 0;j < this->Nx;j++){
                    (*this->boundary)[j+i*Nx] = boundaryIn[j+i*Nx];
                    std::cout<<(*this->boundary)[j+i*Nx]<<" ";
                }
                std::cout<<"\n";
            }std::cout<<std::flush;

            this->cOffset->write_to_device();
            this->weight->write_to_device();
            this->tMatrix->write_to_device();

        }
        double* dirinit(double*dir,double fx, double fy,double d){
            std::cout<<"n"<<std::flush;
            double MachspeedMS = 343;
            double msc = d/MachspeedMS/sqrt(3);
            //filling out density
            for(int i = 0;i < this->Nl;i++){
                dir[i] = d/9;
            }
            //filling out velocity
            for(int i = 0;i < this->Nl;i++){
                if(cy[i]==0)dir[i] += fx/2 * cx[i] * msc;
                if(cx[i]==0)dir[i] += fy/2 * cy[i] * msc;
            }
            std::cout<<"m"<<std::flush;
            return dir;
        }
    public:
        BoltzmanLattice(){
            int bound[this->Nx*this->Ny] = {};
            for(int i = 0;i < this->Nx;i++){
                for(int j = 0;j < this->Ny;j++){
                    bound[i+j*this->Nx] = (this->Nx/4-i)*(this->Nx/4-i) + (this->Ny/2-j)*(this->Ny/2-j) < min(this->Nx,this->Ny)*min(this->Nx,this->Ny)/16;
                }
            }
            double dir[this->Nl] = {0};
            Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device
            this->init(device,bound,dirinit(dir,343,0,avgdensity));
            this->device = device;
        }
        BoltzmanLattice(int* bound,double fx, double fy,double d){
            double dir[this->Nl] = {0};
            Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device
            this->init(device,bound,dirinit(dir,fx,fy,d));
            this->device = device;
        }
        BoltzmanLattice(int* bound,double* airflow){
            Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device
            this->init(device,bound,airflow);
            this->device = device;
        }
        void boundarygen(int* bound){
            for(int j = 1;j < Ny-1;j++){
                for(int i = 1;i < Nx-1;i++){
                    int cnt = 0;
                    for(int k = 0;k < Nl;k++){
                        int offsetx = (*this->cOffset)[k];
                        int offsety = (*this->cOffset)[k+Nl];
                        if(bound[i+(j+offsety)*Nx+offsetx]!=0){
                            cnt++;
                        }
                    }
                    if(cnt == 9){
                        bound[i+(j)*Nx]=2;
                    }
                }
            }
        }
        void tick(){


            Device device = this->device;
            Kernel tick_lattice(this->device, (this->Nx)*(this->Ny), "tick_lattice", *this->Lattice, *this->LatticeEnd,*this->Size,*this->boundary,*this->weight,*this->dAir,*this->cOffset,*this->tMatrix); // kernel that runs on the device
            this->cOffset->write_to_device();
            this->weight->write_to_device();
            this->tMatrix->write_to_device();
            //printArr(Lattice,Nx*Ny*Nl);
            //printArr(LatticeEnd,Nx*Ny*Nl);
            this->Size->write_to_device();
            this->boundary->write_to_device();
            this->Lattice->write_to_device(); // copy data from host memory to device memory


            tick_lattice.run(); // run add_kernel on the device

            this->LatticeEnd->read_from_device();
            //for(int i = 0;i < Nx*Ny*Nl;i++){
            //    Lattice[i] = LatticeEnd[i];
            //}
            this->LatticeEnd->write_to_device();

            Kernel copyarr(device, (this->Nx)*(this->Ny)*(this->Nl), "copyarr", *this->Lattice,*this->LatticeEnd,*this->Size); // kernel that runs on the device
            copyarr.run();
            this->Lattice->read_from_device();
        }
        void printConsole(){
            //printArr((*(this->LatticeEnd)),Nx*Ny*Nl);
            //printCFDDG((*(this->LatticeEnd)));

            /*for(int j = 0;j < Ny;j++){
                for(int i = 0;i < Nx;i++){
                    double d = 0;
                    for(int k = 0;k < Nl;k++){
                        int offsetx = (*this->cOffset)[k];
                        int offsety = (*this->cOffset)[k+Nl];
                        d += (*(this->LatticeEnd))[(j*Nx+i)*Nl+k]*offsety;
                    }
                    std::cout<<gradient[69-std::max(0,std::min(69,int(d/2)+35))];
                } std::cout<<"\n";
            } std::cout<<"\n";*/
            /*for(int j = 0;j < Ny;j++){
                for(int i = 0;i < Nx;i++){
                    double d = 0;
                    for(int k = 0;k < Nl;k++){
                        int offsetx = (*this->cOffset)[k];
                        int offsety = (*this->cOffset)[k+Nl];
                        d += (*(this->LatticeEnd))[(j*Nx+i)*Nl+k]*offsetx;
                    }
                    std::cout<<gradient[69-std::max(0,std::min(69,int(d/avgdensity*200)+35))];
                } std::cout<<"\n";
            } std::cout<<"\n";
            std::cout<<std::flush;*/
            for(int j = 0;j < Ny;j++){
                for(int i = 0;i < Nx;i++){
                    double ax = 0;
                    double ay = 0;
                    double c = 0;
                    for(int k = 0;k < Nl;k++){
                        int offsetx = (*this->cOffset)[k];
                        int offsety = (*this->cOffset)[k+Nl];
                        ax += (*(this->LatticeEnd))[(j*Nx+i)*Nl+k]*offsetx;
                        ay += (*(this->LatticeEnd))[(j*Nx+i)*Nl+k]*offsety;
                        c += (*(this->LatticeEnd))[(j*Nx+i)*Nl+k];
                        //d += sqrt(ax*ax+ay*ay);
                    }
                    double d = sqrt((ax/c)*(ax/c)+(ay/c)*(ay/c));
                    std::cout<<gradient[69-std::max(0,std::min(69,int(d*400)))]<<" ";
                } std::cout<<"\n";
            } std::cout<<"\n";
            std::cout<<std::flush;/**/
        }

};
int* airfoilgen(int* dat, uint Nx,uint Ny){
    for(int j = 0;j < Ny;j++){
        for(int i = 0;i < Nx;i++){
            double x = (double)i/Nx*2-0.1;
            dat[i+j*Nx] = abs((double)(j - ((int)Ny)/2))/Ny <= 5*(0.15)*(0.2969*sqrt(x)-0.126*x-0.3516*x*x+0.2843*x*x*x-0.1015*x*x*x*x);
        }
    }
    return dat;
}

int main() {
    int bound[400*100] = {};

    BoltzmanLattice cfd(airfoilgen(bound,400,100),343,0,1.225);// it starts to breakdown at > mach-1, so DONT DO MACH-1 PLS
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);
    do{
        std::cout<<"Sim\n"<<std::flush;
        cfd.tick();
        cfd.printConsole();
        //wait();
    }while(true);

	return 0;
}

