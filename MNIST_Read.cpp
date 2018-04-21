#include "include/MNIST_Read.h"

//use global vector to store the matrix 
using namespace std;
// BYTE** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
vector<BYTE> read_mnist_images(string full_path) {
    int number_of_images,image_size;//modification
    auto reverseInt = [](int i) {       
        BYTE c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
        // cout << "rows: " << n_rows << "cols: " << n_cols << endl;
        istreambuf_iterator<char> start(file),end;
        vector<BYTE> dataset(start, end);
        return dataset;
        // BYTE** _dataset = new BYTE*[number_of_images];
        // for(int i = 0; i < number_of_images; i++) {
            // _dataset[i] = new BYTE[image_size];
            // file.read((char *)_dataset[i], image_size);
            // cout << Map<Matrix<BYTE,28,28,RowMajor>>((BYTE*)_dataset[i]) << endl;;
        // }
        //Segementation fault here, matrix too big
        // n_rows = n_rows * number_of_images;
        // cout << "p1" << endl;
        // Map<Matrix<BYTE,Dynamic,Dynamic,RowMajor>> MNIST_dataset(*_dataset,n_rows,n_cols);
        // cout << MNIST_dataset.rows();
        // MNIST_dataset.cast<int>();
        // cout << MNIST_dataset.rows() << endl;
        // cout << MNIST_dataset << endl;
        // return MNIST_dataset.cast<int>();
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

//for reading labels dataset
// BYTE* read_mnist_labels(string full_path, int& number_of_labels) {
vector<BYTE> read_mnist_labels(string full_path) {
    int number_of_labels;


    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        // BYTE* _dataset = new BYTE[number_of_labels];
        // for(int i = 0; i < number_of_labels; i++) {
        //     file.read((char*)&_dataset[i], 1);
        // }
        istreambuf_iterator<char> start(file),end;
        vector<BYTE> dataset(start, end);
        return dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}




// int main(int argc,char* argv[]){
//     if (argc!=3){
//         cout << "exactly 3 arguments required." << endl;
//         return 1;
//     }
//     else{
//         if(!strcmp(argv[2],"images")){
//             Matrix<int,Dynamic,Dynamic,RowMajor> dataset;
//             dataset = read_mnist_images(argv[1]);
//             cout << "p3" << endl;
//             cout << dataset.rows() << " " << dataset.cols() <<endl;
//             cout << dataset.topRows(28) <<endl;
//             // Matrix2d m;
//             // m << 2,3,4,5;
//             // cout << m.topRows(2) << endl;
//             // cout << dataset.topRows(28) << endl;
//             // cout << dataset << endl;
//         }
//         else if(!strcmp(argv[2],"labels")){
//             BYTE * labelV;
//             labelV = read_mnist_labels(argv[1]);
//         }
//     }
//     return 0;
// }