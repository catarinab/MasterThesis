

double factorial(double num) {
    if(num == 0)
        return 1;
        
    double res = 1;
    for(int i = 1; i <= num; i++)
        res *= i;
    return res;
}