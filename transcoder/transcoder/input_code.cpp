#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

int main() {
    int t, n;
    cin >> t;
    for (int k = 0; k < t; ++k) {
        int a_left, a_right, a;
        cin >> n;
        for (int i = 0; i < n; ++i) {
            if (i == 0)
                cin >> a_left;
            else if (i == n - 1)
                cin >> a_right;
            else
                cin >> a;
        }
        if (a_left < a_right)
            cout << "YES\n";
        else
            cout << "NO\n";
    }
}
