#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

// mpiCC main.cpp -I /usr/local/include/opencv4 -lopencv_core -lopencv_imgcodecs
// mpirun -np 8 a.out

std::string old_image_name = "horse.ppm";
std::string new_image_name = "output.ppm";
int kernel_size = 27;

using namespace std;
int* compute(int kernel_size, int* mat, int rows, int width)
{
    int* ans = (int *)malloc(sizeof(int)*rows*3*width);
    vector<vector<double>> kernel;
    for(int j = 0; j < kernel_size; j++)
    {
        vector<double> v1(kernel_size, 1.0);
        if(j == kernel_size / 2)
            v1[j] = -1 * (kernel_size * kernel_size - 1);

        kernel.push_back(v1);
    }
    
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0 ; j < width*3 ; j+=3)
        {
            double temp_red = 0, temp_green = 0, temp_blue = 0;
            for (int k = 0; k < kernel.size(); k++)
            {
                for (int l = 0; l < kernel[0].size(); l++)
                {   
                    temp_red += kernel[k][l]*mat[(i+k)*width*3+j+3*l];
                    temp_green += kernel[k][l]*mat[(i+k)*width*3+(j+1)+3*l];
                    temp_blue += kernel[k][l]*mat[(i+k)*width*3+(j+2)+3*l]; 
                }   
            }
            temp_red  = ((temp_red * 255) / ((255*(kernel_size*kernel_size-1))*2)) + 128;
            temp_green = ((temp_green * 255) / ((255*(kernel_size*kernel_size-1))*2)) + 128;
            temp_blue = ((temp_blue * 255) / ((255*(kernel_size*kernel_size-1))*2)) + 128;

            ans[i*width*3 + j  ] = temp_red;
            ans[i*width*3+(j+1)] = temp_green;
            ans[i*width*3+(j+2)] = temp_blue;
        }
    }
    return ans;
}

int main()
{
    int size, rank;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0)
    {
        ////////////////////////// img to .ppm ///////////////////////////
        cv::Mat image = cv::imread("horse.png"); 
        if (image.empty()) 
        {
            cerr << "[!] couldn't open file " << endl;
            return 1;
        }
        std::vector<int> p(2);
        p[0] = cv::IMWRITE_PXM_BINARY;
        p[1] = 0; 
        cv::imwrite("horse.ppm", image, p );

        ////////////////////////// Read Image ////////////////////////////
        string type;
        int width, height, rgb;
        vector<int> mat;

        ifstream old_image;
        old_image.open(old_image_name);

        if(!old_image.is_open())
            cerr << "[!] couldn't open file "<< old_image_name << endl;
    
        old_image >> type;
        old_image >> width;
        old_image >> height;
        old_image >> rgb;

        string red="", green="",blue="";
        int r = 0, g =0, b =0;

        while(!old_image.eof()){

            old_image >> red;
            old_image >> green;
            old_image >> blue;

            stringstream redStream(red);
            stringstream greenStream(green);
            stringstream blueStream(blue);

            redStream >> r;
            greenStream >> g;
            blueStream >> b;
    
            mat.push_back(r);
            mat.push_back(g);
            mat.push_back(b);

        }
        old_image.close();

        /////////////////////////// Padding Zeros //////////////////////////////

        int* arr;
        arr = (int *)malloc(sizeof(int)*(width*3*height+(kernel_size/2)*2*width*3));
        if(arr == NULL) return -30;
        //padding top and bottom
        memset(arr, 0, sizeof(int)*(width*3*height+(kernel_size/2)*2*width*3));
        copy( mat.begin(), mat.end(), &arr[(kernel_size/2)*width*3]);

        ////////////////////////// Distribute Image  ////////////////////////////

        int rowsToSend;
        if(size != 1)  rowsToSend = height / (size-1);
        else rowsToSend = 0;
        int count = rowsToSend*width*3;
        int count_send = count + (kernel_size / 2)*2*width*3;

        bool work = false;
        if(rowsToSend != 0)
            work = true;
        for(int recv = 1; recv < size; recv++)
        {
            MPI_Send(&work, 1, MPI_CXX_BOOL, recv, 0, MPI_COMM_WORLD);
        }

        int* ans = (int *)malloc(sizeof(int)*width*3*height);
        if(rowsToSend!=0)
        {  
            for(int receiver = 1; receiver < size; receiver++)
            {
                MPI_Send(&width, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
                MPI_Send(&rowsToSend, 1, MPI_INT, receiver, 4, MPI_COMM_WORLD);
                MPI_Send(&arr[(receiver-1)*count], count_send, MPI_INT, receiver, 2, MPI_COMM_WORLD);
            }        
            for(int sender = 1; sender < size; sender++)
            {
                MPI_Recv(&ans[(sender-1)*count], count, MPI_INT, sender, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
       
        ////////////////////////// Handle Remainder /////////////////////////
        if(height-(rowsToSend*(size-1))!=0)
        {
            int rows = height-(rowsToSend*(size-1));
            int* curr_ans = compute(kernel_size, &arr[(rowsToSend*(size-1))*width*3], rows, width);
            memcpy(&ans[(rowsToSend*(size-1))*width*3], curr_ans, width*3*sizeof(int)*rows);
        }

        ////////////////////////// Write Image ////////////////////////////

        ofstream newImage;
        newImage.open(new_image_name);

        newImage << type << endl;
        newImage << width << " " << height << endl;
        newImage << rgb << endl;

        for(int row = 0 ; row < height; row ++){
            for (int col = 0; col < width; col++)
            {
                newImage << ans[(row*width*3)+col*3] << " ";
                newImage << ans[(row*width*3)+col*3+1] << " ";
                newImage << ans[(row*width*3)+col*3+2] << " " << endl;
            }
            
        }
        newImage.close();
    }else{
            
        MPI_Status statMat;
        bool work;
        MPI_Recv(&work, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(work)
        {
            int width, rows;
            MPI_Recv(&width, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int count;
            MPI_Probe(0,2,MPI_COMM_WORLD, &statMat);
            MPI_Get_count(&statMat, MPI_INT, &count);
            int* mat;
            mat = (int *)malloc(sizeof(int)*count);
            MPI_Recv(mat,count,MPI_INT,0,2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int* ans = compute(kernel_size, mat, rows, width);
            MPI_Send(ans, rows*3*width, MPI_INT, 0, 3, MPI_COMM_WORLD);
        }      
    }
    MPI_Finalize();
}