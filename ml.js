// Basic Train/Test Split
function trainTestSplit(data, targetCol, testSize = 0.2) {
    let shuffled = [...data].sort(() => 0.5 - Math.random());
    let splitIdx = Math.floor(shuffled.length * (1 - testSize));
    let train = shuffled.slice(0, splitIdx);
    let test = shuffled.slice(splitIdx);

    let X_train = train.map(row => { let r = {...row}; delete r[targetCol]; return r; });
    let y_train = train.map(row => row[targetCol]);
    let X_test = test.map(row => { let r = {...row}; delete r[targetCol]; return r; });
    let y_test = test.map(row => row[targetCol]);

    return { X_train, X_test, y_train, y_test };
}

// MinMaxScaler
class MinMaxScaler {
    constructor() { this.min = {}; this.max = {}; }
    fit(data, columns) {
        columns.forEach(col => {
            let values = data.map(row => row[col]).filter(v => typeof v === 'number' && !isNaN(v));
            if(values.length === 0) return;
            this.min[col] = Math.min(...values);
            this.max[col] = Math.max(...values);
        });
    }
    transform(data, columns) {
        return data.map(row => {
            let res = {};
            columns.forEach(col => {
                if (this.max[col] === this.min[col] || !this.min.hasOwnProperty(col)) {
                    res[col] = 0;
                } else {
                    res[col] = (row[col] - this.min[col]) / (this.max[col] - this.min[col]);
                }
            });
            return res;
        });
    }
}

// Convert Array of Objects to 2D Array
function toMatrix(data, columns) {
    return data.map(row => columns.map(col => row[col] || 0));
}

// Basic KNN implementation
class KNN {
    constructor(k = 3) { this.k = k; }
    fit(X, y) { this.X = X; this.y = y; }
    predict(X_test) {
        return X_test.map(testRow => {
            let distances = this.X.map((trainRow, i) => {
                let dist = 0;
                for (let j = 0; j < trainRow.length; j++) {
                    dist += Math.pow(trainRow[j] - testRow[j], 2);
                }
                return { dist: Math.sqrt(dist), label: this.y[i] };
            });
            distances.sort((a, b) => a.dist - b.dist);
            let neighbors = distances.slice(0, this.k);
            let counts = {};
            let maxCount = 0;
            let predLabel = null;
            neighbors.forEach(n => {
                counts[n.label] = (counts[n.label] || 0) + 1;
                if (counts[n.label] > maxCount) {
                    maxCount = counts[n.label];
                    predLabel = n.label;
                }
            });
            return predLabel;
        });
    }
}

// Basic Gaussian Naive Bayes implementation
class GaussianNB {
    fit(X, y) {
        this.classes = Array.from(new Set(y));
        this.mean = {};
        this.var = {};
        this.priors = {};

        this.classes.forEach(c => {
            let classX = X.filter((_, i) => y[i] === c);
            this.priors[c] = classX.length / X.length;

            let n_features = X[0].length;
            this.mean[c] = new Array(n_features).fill(0);
            this.var[c] = new Array(n_features).fill(0);

            for (let j = 0; j < n_features; j++) {
                let sum = 0;
                classX.forEach(row => sum += row[j]);
                this.mean[c][j] = sum / classX.length;

                let varSum = 0;
                classX.forEach(row => varSum += Math.pow(row[j] - this.mean[c][j], 2));
                this.var[c][j] = varSum / classX.length;
            }
        });
    }
    _gaussianPdf(x, mean, var_) {
        let eps = 1e-9;
        let coeff = 1.0 / Math.sqrt(2 * Math.PI * (var_ + eps));
        let exponent = Math.exp(-Math.pow(x - mean, 2) / (2 * (var_ + eps)));
        return coeff * exponent;
    }
    predict(X_test) {
        return X_test.map(testRow => {
            let maxProb = -Infinity;
            let bestClass = null;

            this.classes.forEach(c => {
                let prob = Math.log(this.priors[c]);
                for (let j = 0; j < testRow.length; j++) {
                    let pdf = this._gaussianPdf(testRow[j], this.mean[c][j], this.var[c][j]);
                    prob += Math.log(pdf + 1e-9);
                }
                if (prob > maxProb) {
                    maxProb = prob;
                    bestClass = c;
                }
            });
            return bestClass;
        });
    }
}

function calculateAccuracy(y_true, y_pred) {
    let correct = 0;
    for (let i = 0; i < y_true.length; i++) {
        if (y_true[i] === y_pred[i]) correct++;
    }
    return correct / y_true.length;
}

function computeConfusionMatrix(y_true, y_pred, numClasses) {
    let matrix = Array(numClasses).fill(0).map(() => Array(numClasses).fill(0));
    for (let i = 0; i < y_true.length; i++) {
        matrix[y_true[i]][y_pred[i]]++;
    }
    return matrix;
}
