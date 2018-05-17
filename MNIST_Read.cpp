#include "include/MNIST_Read.h"

//use global std::vector to store the matrix 
using namespace Eigen;
// BYTE** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {
std::vector<BYTE> read_mnist_images(std::string full_path) {
    int number_of_images,image_size;//modification
    auto reverseInt = [](int i) {       
        BYTE c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);
        // std::cout << "rows: " << n_rows << "cols: " << n_cols << std::endl;
        std::istreambuf_iterator<char> start(file),end;
        std::vector<BYTE> dataset(start, end);
        return dataset;
        // BYTE** _dataset = new BYTE*[number_of_images];
        // for(int i = 0; i < number_of_images; i++) {
            // _dataset[i] = new BYTE[image_size];
            // file.read((char *)_dataset[i], image_size);
            // std::cout << Map<Matrix<BYTE,28,28,RowMajor>>((BYTE*)_dataset[i]) << std::endl;;
        // }
        //Segementation fault here, matrix too big
        // n_rows = n_rows * number_of_images;
        // std::cout << "p1" << std::endl;
        // Map<Matrix<BYTE,Dynamic,Dynamic,RowMajor>> MNIST_dataset(*_dataset,n_rows,n_cols);
        // std::cout << MNIST_dataset.rows();
        // MNIST_dataset.cast<int>();
        // std::cout << MNIST_dataset.rows() << std::endl;
        // std::cout << MNIST_dataset << std::endl;
        // return MNIST_dataset.cast<int>();
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

//for reading labels dataset
// BYTE* read_mnist_labels(std::string full_path, int& number_of_labels) {
std::vector<BYTE> read_mnist_labels(std::string full_path) {
    int number_of_labels;
    std::ifstream file(full_path, std::ios::binary);
    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        // BYTE* _dataset = new BYTE[number_of_labels];
        // for(int i = 0; i < number_of_labels; i++) {
        //     file.read((char*)&_dataset[i], 1);
        // }
        std::istreambuf_iterator<char> start(file),end;
        std::vector<BYTE> dataset(start, end);
        return dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void mnist_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest){
    std::string train_image_path = "train-images";
    std::string train_label_path = "train-labels";
    std::string test_image_path = "test-images";
    std::string test_label_path = "test-labels";
    std::vector<BYTE> train_image_dataset = read_mnist_images(train_image_path);
    std::vector<BYTE> train_label_dataset = read_mnist_labels(train_label_path);
    std::vector<BYTE> test_image_dataset = read_mnist_images(test_image_path);
    std::vector<BYTE> test_label_dataset = read_mnist_labels(test_label_path);
    
    std::vector<double> Xt_train(train_image_dataset.begin(),train_image_dataset.end());
    std::vector<double> y_train(train_label_dataset.begin(),train_label_dataset.end());
    std::vector<double> Xt_test(test_image_dataset.begin(),test_image_dataset.end());
    std::vector<double> y_test(test_label_dataset.begin(),test_label_dataset.end());  
    //classification
    std::vector<double> Xt_train_classify,y_train_classify,Xt_test_classify,y_test_classify;
    for(int i=0;i<y_train.size();++i){
        if(y_train[i]<=1){
            Xt_train_classify.insert(Xt_train_classify.end(),Xt_train.begin()+i*784,Xt_train.begin()+(i+1)*784);
            y_train_classify.push_back(y_train[i]);
        }
    }
    for(int i=0;i<y_test.size();++i){
        if(y_test[i]<=1){
            Xt_test_classify.insert(Xt_test_classify.end(),Xt_test.begin()+i*784,Xt_test.begin()+(i+1)*784);
            y_test_classify.push_back(y_test[i]);
        }
    }
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> Xtt(Xt_train_classify.data(), 784, Xt_train_classify.size()/784);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yy(y_train_classify.data(), y_train_classify.size(), 1);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> XttTest(Xt_test_classify.data(), 784, Xt_test_classify.size()/784);
    Map<Matrix<double,Dynamic,Dynamic,ColMajor>> yyTest(y_test_classify.data(), y_test_classify.size(), 1);
    //normalization
    for(int i=0;i<Xtt.cols();++i){
        Xtt.col(i) = Xtt.col(i)/Xtt.col(i).norm();
    }
    for(int i=0;i<XttTest.cols();++i){
        XttTest.col(i) = XttTest.col(i)/XttTest.col(i).norm();
    }
    std::cout << "Xt rows: " << Xtt.rows() << std::endl << "Xt cols: " << Xtt.cols() << std::endl;
    std::cout << "XtTest rows: " << XttTest.rows() << std::endl << "XtTest cols: " << XttTest.cols() << std::endl;
    std::cout << "y cols: " << yy.size() << std::endl;
    std::cout << "yTest cols: " << yyTest.size() << std::endl;
    Xt = Xtt; y = yy; XtTest = XttTest; yTest = yyTest;
}



// int main(int argc,char* argv[]){
//     if (argc!=3){
//         std::cout << "exactly 3 arguments required." << std::endl;
//         return 1;
//     }
//     else{
//         if(!strcmp(argv[2],"images")){
//             Matrix<int,Dynamic,Dynamic,RowMajor> dataset;
//             dataset = read_mnist_images(argv[1]);
//             std::cout << "p3" << std::endl;
//             std::cout << dataset.rows() << " " << dataset.cols() <<std::endl;
//             std::cout << dataset.topRows(28) <<std::endl;
//             // Matrix2d m;
//             // m << 2,3,4,5;
//             // std::cout << m.topRows(2) << std::endl;
//             // std::cout << dataset.topRows(28) << std::endl;
//             // std::cout << dataset << std::endl;
//         }
//         else if(!strcmp(argv[2],"labels")){
//             BYTE * labelV;
//             labelV = read_mnist_labels(argv[1]);
//         }
//     }
//     return 0;
// }