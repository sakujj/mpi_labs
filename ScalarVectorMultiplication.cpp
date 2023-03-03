#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <mpi.h>

using namespace std;

void fillInWithRandomInts(int* array, int len, int mod, bool isNegative) {
    int mult = isNegative ? -1 : 1;
    for (int i = 0; i < len; ++i) {
        array[i] = mult * (rand() % mod);
    }
}

void coutIntVector(int* vector, int len) {
    cout << "(";
    for (int i = 0; i < len; ++i) {
        printf("%2d", vector[i]);
        if (i != len - 1) {
            std::cout << ", ";
        } else {
            std::cout << ")\n";
        }
    }
}


int main(int argc, char** argv) {
    int root = 0;
    int len;
    sscanf(argv[1], "%d", &len);
    int prcsCount;
    int rank;
    MPI_Datatype datatype = MPI_INT;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &prcsCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm;
    if (len / prcsCount == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, rank / len, rank, &comm);
        if (rank >= len) {
            MPI_Finalize();
            return 0;
        }

        prcsCount = len;
    } else {
        comm = MPI_COMM_WORLD;
    }

    int* rootRecvBuf = nullptr;
    int* rootRecvCounts = nullptr;
    int* rootDispls = nullptr;

    int* vector_1 = nullptr;
    int* vector_2 = nullptr;
    int* sendCounts = nullptr;
    int* displs = nullptr;

    int* recvBuf_1 = nullptr;
    int* recvBuf_2 = nullptr;

    int sendCount;
    int recvCount;

    sendCount = len / prcsCount;
    if (len % prcsCount != 0 && rank < len % prcsCount) {
        sendCount += 1;
    }

    recvCount = sendCount;
    recvBuf_1 = new int[recvCount];

    if (rank == root) {
        vector_1 = new int[len];
        vector_2 = new int[len];
        fillInWithRandomInts(vector_1, len, 2, true);
        cout << "\n";
        coutIntVector(vector_1, len);
        cout << "\n";
        fillInWithRandomInts(vector_2, len, 2, true);
        coutIntVector(vector_2, len);
        cout << endl;

        sendCounts = new int[prcsCount];
        displs = new int[prcsCount];
        displs = new int[prcsCount];
        if (len % prcsCount == 0) {
            for (int i = 0; i < prcsCount; ++i) {
                sendCounts[i] = sendCount;
                displs[i] = sendCounts[i] * i;
            }
        } else {
            int minCount = len % prcsCount;
            for (int i = 0; i < prcsCount; ++i) {
                if (i < minCount) {
                    sendCounts[i] = sendCount;
                    displs[i] = sendCounts[i] * i;
                } else {
                    sendCounts[i] = sendCount - 1;
                    displs[i] = sendCounts[i] * i + minCount;
                }
            }
        }
    }

    MPI_Barrier(comm);

    MPI_Scatterv(vector_1, sendCounts, displs, datatype,
        recvBuf_1, recvCount, datatype,
        root, comm);

    MPI_Scatterv(vector_2, sendCounts, displs, datatype,
        recvBuf_2, recvCount, datatype,
        root, comm);

    int result = 0;
    for (int i = 0; i < recvCount; ++i) {
        result += recvBuf_1[i] * recvBuf_2[i];
    }

    cout << "    " << rank << "     process got    " << recvCount << "    coordinates to mulptiply. Result is    " << result << endl;
    MPI_Barrier(comm);


    if (rank == root) {
        rootRecvBuf = new int[prcsCount];
        rootRecvCounts = new int[prcsCount];
        rootDispls = new int[prcsCount];
        for (int i = 0; i < prcsCount; ++i) {
            rootRecvCounts[i] = 1;
            rootDispls[i] = i * 1;
        }
    }

    MPI_Gatherv(&result, 1, datatype,
        rootRecvBuf, rootRecvCounts, rootDispls, datatype,
        root, comm);
    int product = 0;
    if (rank == root) {
        for (int i = 0; i < prcsCount; ++i) {
            product += rootRecvBuf[i];
        }
        cout << "\n-----------------------------------------------\n";
        cout << "-----------------------------------------------\n\n";

        cout << "The scalar product of two vectors equals: " << product << endl;
        cout << "\n-----------------------------------------------\n";
        cout << "-----------------------------------------------\n";
    }

    MPI_Finalize();

}

