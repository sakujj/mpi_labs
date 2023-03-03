#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <mpi.h>

using namespace std;

void fillInWithRandomInts(int* array, int len, int mod) {
    for (int i = 0; i < len; ++i) {
        array[i] = (rand() % mod + 1);
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
    int root = 0;
    int MOD = 100;
    int m;
    int n;
    sscanf(argv[1], "%d", &m);
    sscanf(argv[2], "%d", &n);

    int prcsCount;
    int rank;
    MPI_Datatype datatype = MPI_INT;
    MPI_Init(&argc, &argv);


    MPI_Comm_size(MPI_COMM_WORLD, &prcsCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    if (m / prcsCount == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, rank / m, rank, &comm);
        if (rank >= m) {
            MPI_Finalize();
            return 0;
        }

        prcsCount = m;
    } else {
        comm = MPI_COMM_WORLD;
    }

    int* rootRecvBuf = nullptr;
    int* rootRecvCounts = nullptr;
    int* rootDispls = nullptr;

    int* matrix_1 = nullptr;
    int* matrix_2 = nullptr;
    int* sendCounts = nullptr;
    int* displs = nullptr;

    int* recvBuf_1 = nullptr;
    int* recvBuf_2 = nullptr;

    int sendCount;
    int recvCount;

    sendCount = m / prcsCount;
    if (m % prcsCount != 0 && rank < m % prcsCount) {
        sendCount += 1;
    }

    recvCount = sendCount;
    recvBuf_1 = new int[recvCount * n];
    recvBuf_2 = new int[recvCount * n];

    if (rank == root) {
        matrix_1 = new int[m * n];
        matrix_2 = new int[m * n];
        fillInWithRandomInts(matrix_1, m * n, MOD);
        cout << "\n";
        coutNaturalMatrix(matrix_1, m, n, MOD);
        cout << "\n";
        fillInWithRandomInts(matrix_2, m * n, MOD);
        cout << "\n";
        coutNaturalMatrix(matrix_2, m, n, MOD);
        cout << endl;

        sendCounts = new int[prcsCount];
        displs = new int[prcsCount];
        if (m % prcsCount == 0) {
            for (int i = 0; i < prcsCount; ++i) {
                sendCounts[i] = sendCount * n;
                displs[i] = sendCounts[i] * i;
            }
        } else {
            int redundant = m % prcsCount;
            for (int i = 0; i < prcsCount; ++i) {
                if (i < redundant) {
                    sendCounts[i] = sendCount * n;
                    displs[i] = sendCounts[i] * i;
                } else {
                    sendCounts[i] = (sendCount - 1) * n;
                    displs[i] = sendCounts[i] * i + redundant * n;
                }
            }
        }
    }

    MPI_Barrier(comm);

    MPI_Scatterv(matrix_1, sendCounts, displs, datatype,
        recvBuf_1, recvCount * n, datatype,
        root, comm);

    MPI_Scatterv(matrix_2, sendCounts, displs, datatype,
        recvBuf_2, recvCount * n, datatype,
        root, comm);

    int* resultRows = new int[recvCount * n];
    for (int i = 0; i < recvCount * n; ++i) {
        resultRows[i] = recvBuf_1[i] - recvBuf_2[i];
    }

    cout << "    " << rank << "     process got    " << recvCount << "    rows to perform subtraction.\n";
    cout << " Subtract from: \n";
    coutNaturalMatrix(recvBuf_1, recvCount, n, MOD);
    cout << " Subtracted: \n";
    coutNaturalMatrix(recvBuf_2, recvCount, n, MOD);
    cout << " Result is    \n";
    coutNaturalMatrix(resultRows, recvCount, n, MOD);
    cout << "\n\n";
    MPI_Barrier(comm);


    if (rank == root) {
        rootRecvBuf = new int[m * n];
        rootRecvCounts = sendCounts;
        rootDispls = displs;
    }

    MPI_Gatherv(resultRows, recvCount * n, datatype,
        rootRecvBuf, rootRecvCounts, rootDispls, datatype,
        root, comm);

    if (rank == root) {
       
        cout << "\n-----------------------------------------------\n";
        cout << "-----------------------------------------------\n\n";

        cout << "   The result of subtraction of two matrices: \n";
        coutNaturalMatrix(matrix_1, m, n, MOD);
        cout << "\n";
        cout << " and(minus) \n";
        cout << "\n";
        coutNaturalMatrix(matrix_2, m, n, MOD);
        cout << "   is equals to: \n";
        coutNaturalMatrix(rootRecvBuf, m, n, MOD);
        cout << "\n-----------------------------------------------\n";
        cout << "-----------------------------------------------\n";
    }

    MPI_Finalize();

}

