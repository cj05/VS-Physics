#include "BML.cpp"



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

