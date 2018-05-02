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

void mnist_read(MatrixXd &Xt, VectorXd &y, MatrixXd &XtTest, VectorXd &yTest){
    string train_image_path = "train-images";
    string train_label_path = "train-labels";
    string test_image_path = "test-images";
    string test_label_path = "test-labels";
    vector<BYTE> train_image_dataset = read_mnist_images(train_image_path);
    vector<BYTE> train_label_dataset = read_mnist_labels(train_label_path);
    vector<BYTE> test_image_dataset = read_mnist_images(test_image_path);
    vector<BYTE> test_label_dataset = read_mnist_labels(test_label_path);
    
    vector<double> Xt_train(train_image_dataset.begin(),train_image_dataset.end());
    vector<double> y_train(train_label_dataset.begin(),train_label_dataset.end());
    vector<double> Xt_test(test_image_dataset.begin(),test_image_dataset.end());
    vector<double> y_test(test_label_dataset.begin(),test_label_dataset.end());  
    //classification
    vector<double> Xt_train_classify,y_train_classify,Xt_test_classify,y_test_classify;
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
    cout << "Xt rows: " << Xtt.rows() << endl << "Xt cols: " << Xtt.cols() << endl;
    cout << "XtTest rows: " << XttTest.rows() << endl << "XtTest cols: " << XttTest.cols() << endl;
    cout << "y cols: " << yy.size() << endl;
    cout << "yTest cols: " << yyTest.size() << endl;
    Xt = Xtt; y = yy; XtTest = XttTest; yTest = yyTest;
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