#include "opencl.hpp"
#include "iostream"
#define printArr(n,N) for(int i = 0;i < N/Nx/3;i++) {for(int j = 0;j < Nx*3;j++) std::cout<<n[(i%3*3+j%3) + (j/3*9+i/3*Nx*9)]<<" "; std::cout<<"\n";} std::cout<<"\n"; std::cout<<std::flush;

#define printCFDDG(n) for(int j = 0;j < Ny;j++) {for(int i = 0;i < Nx;i++){ double d = 0; for(int k = 0;k < Nl;k++) d += n[(j*Nx+i)*Nl+k]; std::cout<<gradient[69-std::min(69,int(d/3))]<<" ";} std::cout<<"\n";} std::cout<<"\n"; std::cout<<std::flush;

#define printCFDD(n) for(int j = 0;j < Ny;j++) {for(int i = 0;i < Nx;i++){ double d = 0; for(int k = 0;k < Nl;k++) d += n[(j*Nx+i)*Nl+k]; std::cout<<int(d)<<" ";} std::cout<<"\n";} std::cout<<"\n"; std::cout<<std::flush;
//

class BoltzmanLattice{
    char gradient[] = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
    private:
        const uint Nx = 20;
        const uint Ny = 10;
        const uint Nl = 9;
        double avgdensity  = 100;    // average density
        double tau         = 0.6;    // collision timescale
        Device device;
        Memory<double> *Lattice;
        Memory<double> *LatticeEnd;
        Memory<int> *boundary;
        Memory<double> *weight;
        Memory<double> *Size;
        Memory<double> *dAir;

        double* w;

        void init(Device device,int* boundaryIn,double* enviromentAir){
            double density = 0;
            for(int i = 0; i < Nl;i++){
                density += enviromentAir[i];
            }
            for(int i = 0; i < Nl;i++){
                enviromentAir[i]*=this->avgdensity/density/2;
            }

            this->Lattice = new Memory<double>(device, this->Nx*this->Ny*this->Nl); // 3darray in 1d, access data at l+(y+x*Ny)*Nl
            this->LatticeEnd = new Memory<double>(device, this->Nx*this->Ny*this->Nl); // 3darray in 1d, access data at l+(y+x*Ny)*Nl
            this->boundary = new Memory<int>(device, this->Nx*this->Ny);
            this->weight = new Memory<double>(device, Nl);
            this->Size = new Memory<double>(device,4);
            this->dAir = new Memory<double>(device,Nl,1,enviromentAir);

            const double w[] = {1/36.,1/9.,1/36.,1/9.,4/9.,1/9.,1/36.,1/9.,1/36.};
            for(int i = 0; i < (this->Nl);i++){
                (*this->weight)[i] = w[i];
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


            for(int i = 0;i < this->Ny; i++){
                for(int j = 0;j < this->Nx;j++){
                    (*this->boundary)[j+i*Nx] = boundaryIn[j+i*Nx];
                    std::cout<<(*this->boundary)[j+i*Nx]<<" ";
                }
                std::cout<<"\n";
            }std::cout<<std::flush;

        }

    public:
        BoltzmanLattice(){
            Device device(select_device_with_most_flops()); // compile OpenCL C code for the fastest available device
            double dir[9] = {1,1,1,1,1,2,1,1,1};
            int bound[this->Nx*this->Ny] =  {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,1,2,2,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,1,1,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                            ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            this->init(device,bound,dir);
            this->device = device;
        }


        void tick(){


            Device device = this->device;
            Kernel tick_lattice(this->device, (this->Nx)*(this->Ny), "tick_lattice", *this->Lattice, *this->LatticeEnd,*this->Size,*this->boundary,*this->weight,*this->dAir); // kernel that runs on the device
            this->weight->write_to_device();
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
            printCFDDG((*(this->LatticeEnd)));
        }

};
