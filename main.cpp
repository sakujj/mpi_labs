#define _CRT_SECURE_NO_WARNINGS

#include "Windows.h"
#include <fstream>

using namespace std;

int main() {
    int m = 3000;
    int n = 8500;
    int k = 3500;
    int prcsCount = 8;
    ofstream out("output.txt");
    out.close();


    for (; prcsCount > 0; prcsCount--) {
        char szCmd[256];
        sprintf(szCmd, "mpiexec.exe -n %d matrix_multiplication.exe %d %d %d", prcsCount, m, n, k);
       /* char szDirectory[256] = "C:\\Users\\avata\\source\\repos\\collectStatistics\\x64\\Debug";*/

        STARTUPINFOA si = { sizeof(STARTUPINFOA) };
        PROCESS_INFORMATION pi;
        CreateProcessA(
            nullptr,
            szCmd,
            nullptr,
            nullptr,
            false,
            CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &si,
            &pi
        );
        WaitForSingleObject(pi.hProcess, INFINITE);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }
}