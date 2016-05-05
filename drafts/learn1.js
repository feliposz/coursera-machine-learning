// a very simple, possibly inneficient and naive implementation of univariate linear regression using gradient descent

var fs = require('fs');

// display iterations
var debug = false;

// "learn rate"
var alfa = 0.1;

// "max iterations"
var maxIterations = 100000;

var main = function() {
    debug = false;
    alfa = 0.01;
    maxIterations = 10000;

    //doTest();
    doFileData('data_test3.tsv');
    //doFileData('data_test2.tsv');
    //doFileData('data_test.tsv');
    //doFileData('data_test4.tsv');
    //doFileData('data.tsv');
};

var square = function(x) {
    return x*x;
};

// J(t0, t1) = 1/(2*m) * sum((h(x[i]) - y[i])^2) where h(x) = t0 + t1 * x
var costFunction = function(theta0, theta1, xs, ys) {
    var m = xs.length;
    var sum = 0;
    var i;
    for (i = 0; i < m; i++) {
        sum += square(theta0 + theta1 * xs[i] - ys[i]);
    }
    return sum/2/m;
};

// d/dx of J(t0, t1) on t0 = 1/*m * sum(h(x[i]) - y[i])
var costDerivTheta0 = function(theta0, theta1, xs, ys) {
    var m = xs.length;
    var sum = 0;
    var i;
    for (i = 0; i < m; i++) {
        sum += theta0 + theta1 * xs[i] - ys[i];
    }
    return sum/m;
};

// d/dx of J(t0, t1) on t1 = 1/*m * sum((h(x[i]) - y[i]) * x[i])
var costDerivTheta1 = function(theta0, theta1, xs, ys) {
    var m = xs.length;
    var sum = 0;
    var i;
    for (i = 0; i < m; i++) {
        sum += (theta0 + theta1 * xs[i] - ys[i]) * xs[i];
    }
    return sum/m;
};

var gradientDescentLinearRegression = function (xs, ys) {
    var temp0, temp1;
    var theta0 = 0;
    var theta1 = 0;
    var m = xs.length;
    var iteration = 0;
    var previousCost = costFunction(theta0, theta1, xs, ys);
    while (true) {
        currentCost = costFunction(theta0, theta1, xs, ys);
        if (debug) {
            console.log('iteration = ' + iteration + ', theta0 = ' + theta0 + ', theta1 =' + theta1, ' currentCost = ' + currentCost);
        }
        temp0 = theta0 - alfa * costDerivTheta0(theta0, theta1, xs, ys);
        temp1 = theta1 - alfa * costDerivTheta1(theta0, theta1, xs, ys);
        if (currentCost > previousCost) {
            console.log('diverging!');
            break;
        }
        // simultaneous update
        theta0 = temp0;
        theta1 = temp1;
        iteration++;
        previousCost = currentCost;
        if (iteration > maxIterations) {
            console.log('maxIterations = ' + maxIterations);
            break;
        }
    }

    return {
        theta0: theta0,
        theta1: theta1
    };
};

var calcResult = function(xs, ys) {
    var result = gradientDescentLinearRegression(xs, ys);
    console.log('result: theta0 = ' + result.theta0 + ', theta1 = ' + result.theta1);
};

var doTest = function() {
    var xs, ys;
    //xs = [1, 2, 3, 4, 5];
    //ys = [-1, -3, -5, -7, -9];
    //ys = [3, 6, 7, 10, 13];
    //ys = [2, 4, 6, 8, 10];
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    ys = [27, 43, 54, 49, 71, 66, 87, 90, 115, 105, 116, 129];

    calcResult(0.01, xs, ys);
};

var doFileData = function(file) {
    fs.readFile(file, function (err, data) {
        var lines, xs, ys, i, values, x, y
        console.log(file);
        if (err) {
            console.log(err);
        }
        lines = data.toString().trim().split('\r\n');
        xs = [];
        ys = [];
        for (i = 0; i < lines.length; i++) {
            values = lines[i].split('\t');
            x = parseFloat(values[0]);
            y = parseFloat(values[1]);
            xs.push(x);
            ys.push(y);
        }

        calcResult(xs, ys);
    });
};

main();
