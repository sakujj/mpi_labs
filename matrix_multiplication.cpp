#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <mpi.h>
#include <fstream>

using namespace std;

void fillInWithRandomInts(int* array, int len, int mod) {
    for (int i = 0; i < len; ++i) {
        array[i] = (rand() % mod + 1);
    }
}

int* transposeMatrix(int* array, int m, int n) {
    int* result = new int[m * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            result[i * m + j] = array[j * n + i];
        }
    }

    return result;
}

void fillMatrix(int* array, int len) {
    for (int i = 0; i < len; ++i) {
        int mult = rand() % 2 == 1 ? -1 : 1;
        array[i] = (rand() % 2 * mult);
    }
}

void coutNaturalMatrix(int* matrix, int m, int n, int mod) {
    int width = 1;
    while (mod != 0) {
        mod /= 10;
        ++width;
    }

    for (int j = 0; j < m; ++j) {
        std::cout << "||| ";
        for (int i = 0; i < n; ++i) {
            printf("%*d", width, matrix[i + j * n]);
            std::cout << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    int ROOT = 0;
    int MOD = 2;
    MPI_Datatype DATATYPE = MPI_INT;
    int m;
    int n;
    int k;
    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &n);
    sscanf(argv[3], "%d", &k);

    int prcsCount;
    int rank;
    MPI_Init(&argc, &argv);


    MPI_Comm_size(MPI_COMM_WORLD, &prcsCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    if (n / prcsCount == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, rank / n, rank, &comm);
        if (rank >= n) {
            MPI_Finalize();
            return 0;
        }

        prcsCount = n;
    } else {
        comm = MPI_COMM_WORLD;
    }

    int* rootRecvBuf = nullptr;
    int* rootRecvCounts = nullptr;
    int* rootDispls = nullptr;

    int* matrix_1 = nullptr;
    int* matrix_1_transposed = nullptr;
    int* matrix_2 = nullptr;
    int* sendCounts_1 = nullptr;
    int* sendCounts_2 = nullptr;
    int* displs_1 = nullptr;
    int* displs_2 = nullptr;

    int* recvBuf_1 = nullptr;
    int* recvBuf_2 = nullptr;

    int sendCount;
    int recvCount;

    sendCount = n / prcsCount;
    if (n % prcsCount != 0 && rank < n % prcsCount) {
        sendCount += 1;
    }

    recvCount = sendCount;
    recvBuf_1 = new int[recvCount * m];
    recvBuf_2 = new int[recvCount * k];

    if (rank == ROOT) {
        matrix_1 = new int[m * n];
        matrix_2 = new int[n * k];
        fillMatrix(matrix_1, m * n);
        fillMatrix(matrix_2, n * k);

        //cout << "\n";
        //coutNaturalMatrix(matrix_1, m, n, MOD);
        //cout << "\n";
        //cout << "\n";
        //coutNaturalMatrix(matrix_2, n, k, MOD);
        //cout << endl;
        matrix_1_transposed = transposeMatrix(matrix_1, m, n);
    }

    double time = MPI_Wtime();

    if (rank == ROOT) {
        sendCounts_1 = new int[prcsCount];
        displs_1 = new int[prcsCount];
        sendCounts_2 = new int[prcsCount];
        displs_2 = new int[prcsCount];
        if (n % prcsCount == 0) {
            for (int i = 0; i < prcsCount; ++i) {
                sendCounts_1[i] = sendCount * m;
                displs_1[i] = sendCounts_1[i] * i;

                sendCounts_2[i] = sendCount * k;
                displs_2[i] = sendCounts_2[i] * i;
            }
        } else {
            int redundant = n % prcsCount;
            for (int i = 0; i < prcsCount; ++i) {
                if (i < redundant) {
                    sendCounts_1[i] = sendCount * m;
                    displs_1[i] = sendCounts_1[i] * i;

                    sendCounts_2[i] = sendCount * k;
                    displs_2[i] = sendCounts_2[i] * i;
                } else {
                    sendCounts_1[i] = (sendCount - 1) * m;
                    displs_1[i] = sendCounts_1[i] * i + redundant * m;

                    sendCounts_2[i] = (sendCount - 1) * k;
                    displs_2[i] = sendCounts_2[i] * i + redundant * k;
                }
            }
        }
    }

    MPI_Scatterv(matrix_1_transposed, sendCounts_1, displs_1, DATATYPE,
        recvBuf_1, recvCount * m, DATATYPE,
        ROOT, comm);


    MPI_Scatterv(matrix_2, sendCounts_2, displs_2, DATATYPE,
        recvBuf_2, recvCount * k, DATATYPE,
        ROOT, comm);

    if (rank == ROOT) {
        delete[] matrix_1_transposed;
        delete[] sendCounts_1;
        delete[] sendCounts_2;
        delete[] displs_1;
        delete[] displs_2;
    }


    int* resultPart = new int[m * k];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < m; ++j) {
            resultPart[j * k + i] = recvBuf_2[i] * recvBuf_1[j];
        }
    }
    for (int ii = 1; ii < recvCount; ++ii) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                resultPart[j * k + i] += recvBuf_2[i + ii * k] * recvBuf_1[j + ii * m];
            }
        }
    }

    if (rank == ROOT) {
        rootRecvBuf = new int[m * k];
    }

    MPI_Reduce(resultPart, rootRecvBuf, m * k, DATATYPE, MPI_SUM, ROOT, comm);

    time = MPI_Wtime() - time;

    //if (rank == ROOT) {
    //    cout << "\n-----------------------------------------------\n";
    //    cout << "-----------------------------------------------\n\n";
    //    cout << "   The result of multiplication of two matrices: \n";
    //    coutNaturalMatrix(rootRecvBuf, m, k, MOD);
    //    cout << "\n-----------------------------------------------\n";
    //    cout << "-----------------------------------------------\n";
    //}
    double avgTime;
    MPI_Reduce(&time, &avgTime, 1, MPI_DOUBLE, MPI_SUM, ROOT, comm);

    if (rank == ROOT) {
        avgTime /= prcsCount;
        char* resultingTime = new char[40];
        sprintf(resultingTime, "%d %f", prcsCount, avgTime);

        fstream out("output.txt", ios::app);
        out << resultingTime << "\n";
        out.close();
    }

    delete[] recvBuf_1;
    delete[] recvBuf_2;
    delete[] resultPart;

    if (rank == ROOT) {
        delete[] rootRecvBuf;
        delete[] rootRecvCounts;
        delete[] rootDispls;
        delete[] matrix_1;
        delete[] matrix_2;
    }

    MPI_Finalize();

}

