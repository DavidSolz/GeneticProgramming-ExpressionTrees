#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <vector>

// Definition of abstract class which represents expression
template<typename Type>
class Expression{
public:
    Expression<Type> *left;
    Expression<Type> *right;

    Expression(){
        this->left = nullptr;
        this->right = nullptr;
    }

    Expression(Expression<Type> *l, Expression<Type> *r){
        this->left = l;
        this->right = r;
    }

    virtual Type Evaluate() = 0;

    virtual Expression<Type>* Clone() const = 0;

    virtual void Show() const = 0;

    virtual bool IsValid(){
        return left != nullptr && right != nullptr;
    };

    void Prune() {

        if(left == nullptr && right==nullptr)
            return;

        // Randomly prune one of the children
        if (rand() % 2 == 0) {
            delete left;
            left = nullptr;
        } else {
            delete right;
            right = nullptr;
        }
    }

    ~Expression(){
        if(left != nullptr)
            delete left;

        if(right != nullptr)
            delete right;
        
    }
};

template<typename Type>
class ValueExpression : public Expression<Type>{
public:

    Type value;

    ValueExpression() : Expression<Type>(nullptr, nullptr){}

    ValueExpression(const Type& _value){
        this->value = _value;
        this->left = nullptr;
        this->right = nullptr;
    }

    Expression<Type>* Clone() const override {
        return new ValueExpression<Type>(*this);
    }

    Type Evaluate(){
        return value;
    }

    void Show() const override{
        fprintf(stdout, "%lf ", value);
    }

    bool IsValid() override{
        return this->left==nullptr && this->right==nullptr;
    }
 
};

template<typename Type>
class AddExpression : public Expression<Type>{
public:   

    AddExpression() : Expression<Type>(nullptr, nullptr){}

    AddExpression(Expression<Type>* l, Expression<Type>* r) : Expression<Type>(l, r) {}

    Type Evaluate(){
        return this->left->Evaluate() + this->right->Evaluate();
    }

    void Show() const override{
        fprintf(stdout, "( ");
        this->left->Show();
        fprintf(stdout, "+ ");
        this->right->Show();
        fprintf(stdout, ") ");
    }

    Expression<Type>* Clone() const override {
        Expression<Type> *new_left = this->left->Clone(); 
        Expression<Type> *new_right = this->right->Clone();
        return new AddExpression<Type>(new_left, new_right);
    }
};

template<typename Type>
class SubstractExpression : public Expression<Type>{
public:

    SubstractExpression() : Expression<Type>(nullptr, nullptr){}

    SubstractExpression(Expression<Type>* l, Expression<Type>* r) : Expression<Type>(l, r) {}

    Type Evaluate(){
        return this->left->Evaluate() - this->right->Evaluate();
    }

    void Show() const override{
        fprintf(stdout, "( ");
        this->left->Show();
        fprintf(stdout, "- ");
        this->right->Show();
        fprintf(stdout, ") ");
    }

    Expression<Type>* Clone() const override {
        Expression<Type> *new_left = this->left->Clone(); 
        Expression<Type> *new_right = this->right->Clone();
        return new SubstractExpression<Type>(new_left, new_right);
    }
};

template<typename Type>
class MultiplyExpression : public Expression<Type>{
public:

    MultiplyExpression() : Expression<Type>(nullptr, nullptr){}

    MultiplyExpression(Expression<Type>* l, Expression<Type>* r) : Expression<Type>(l, r) {}

    Type Evaluate(){
        return this->left->Evaluate() * this->right->Evaluate();
    }

    void Show() const override{
        fprintf(stdout, "( ");
        this->left->Show();
        fprintf(stdout, "* ");
        this->right->Show();
        fprintf(stdout, ") ");
    }

    Expression<Type>* Clone() const override {
        Expression<Type> *new_left = this->left->Clone(); 
        Expression<Type> *new_right = this->right->Clone();
        return new MultiplyExpression<Type>( new_left, new_right);
    }
};

template<typename Type>
class DivideExpression : public Expression<Type>{
public:

    DivideExpression() : Expression<Type>(nullptr, nullptr){}

    DivideExpression(Expression<Type>* l, Expression<Type>* r) : Expression<Type>(l, r) {}

    Type Evaluate(){
        return this->left->Evaluate() / this->right->Evaluate();
    }

    void Show() const override{
        fprintf(stdout, "( ");
        this->left->Show();
        fprintf(stdout, "/ ");
        this->right->Show();
        fprintf(stdout, ") ");
    }

    Expression<Type>* Clone() const override {
        Expression<Type> *new_left = this->left->Clone(); 
        Expression<Type> *new_right = this->right->Clone();
        return new DivideExpression<Type>(new_left, new_right);
    }
};

template<typename Type>
Expression<Type>* GenerateRandomExpressionTree(const size_t& maxDepth = 3, const double& pruneProbability = 0.0f) {
    
    if (maxDepth <= 0 || (rand() % 100) < (pruneProbability * 100)) {
        return new ValueExpression<Type>( rand()%255 - 128);
    }

    Expression<Type> * expression;


    switch (rand()%4){
    case 0:
        expression = new AddExpression<Type>();
        break;

    case 1:
        expression = new SubstractExpression<Type>();
        break;

    case 2:
        expression = new MultiplyExpression<Type>();
        break;

    case 3:
        expression = new DivideExpression<Type>();
        break;
    }

    expression->left = GenerateRandomExpressionTree<Type>(maxDepth - 1, pruneProbability);
    expression->right = GenerateRandomExpressionTree<Type>(maxDepth - 1, pruneProbability);

    return expression;
}

template<typename Type>
double EvaluateFitness(Expression<Type>* tree, Type target) {
    Type result = tree->Evaluate();
    return 1.0 / (1.0 + std::abs(result - target));
}

// Function to select the top individuals based on fitness
template<typename Type>
void SelectTopIndividuals(const std::vector<std::pair<Expression<Type>*, double>>& population,
                          std::vector<Expression<Type>*>& selectedPopulation,
                          double selectionPercentage) {
    selectedPopulation.clear();

    // Sort individuals based on fitness (higher fitness is better)
    std::vector<std::pair<Expression<Type>*, double>> sortedPopulation = population;
    std::sort(sortedPopulation.begin(), sortedPopulation.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Select the top individuals based on the selection percentage
    size_t selectedCount = static_cast<size_t>(population.size() * selectionPercentage);
    selectedPopulation.reserve(selectedCount);
    for (size_t i = 0; i < selectedCount; ++i) {
        selectedPopulation.push_back(sortedPopulation[i].first->Clone()); // Clone the selected individuals
    }
}


// Function to mutate a subtree in an expression tree
template<typename Type>
void MutateSubtree(Expression<Type>* tree, const size_t & maxDepth = 2, const double& prunningProbability = 0.0f) {
    // For simplicity, let's randomly replace a leaf node with a new random expression
    if (rand() % 2 == 0) {
        delete tree->left;
        tree->left = GenerateRandomExpressionTree<Type>(maxDepth, prunningProbability);  // Adjust the depth as needed
    } else {
        delete tree->right;
        tree->right = GenerateRandomExpressionTree<Type>(maxDepth, prunningProbability); // Adjust the depth as needed
    }
}

// Function to perform reproduction by crossover
template<typename Type>
Expression<Type>* Crossover(const Expression<Type>* parent1, const Expression<Type>* parent2) {
    // Clone the parents
    Expression<Type>* child1 = parent1->Clone();
    Expression<Type>* child2 = parent2->Clone();

    // Perform crossover (swap subtrees)
    std::swap(child1, child2);

    Expression<Type>* selected = (rand() % 2 == 0) ? child1 : child2;

    if( selected == child1){
        delete child2;
    }else{
        delete child1;
    }

    // Randomly choose which child to return
    return selected;
}

// Function to perform reproduction by mutation
template<typename Type>
Expression<Type>* Mutate(const Expression<Type>* parent, const size_t & maxDepth = 2, const double& prunningProbability = 0.0f) {
    // Clone the parent
    Expression<Type>* child = parent->Clone();

    // Perform mutation (modify a subtree)
    MutateSubtree(child, maxDepth, prunningProbability);

    return child;
}

template<typename Type>
std::vector< std::pair<Expression<Type>*, double> > GeneratePopulation(const Type& target, const size_t& populationSize, const size_t & maxDepth, const double& prunningProbability){
    std::vector< std::pair<Expression<Type>*, double> > population;

    for (size_t i = 0; i < populationSize; ++i) {
        Expression<Type>* tree = GenerateRandomExpressionTree<Type>(maxDepth, prunningProbability);
        double fitness = EvaluateFitness(tree, target);
        population.emplace_back( std::make_pair( tree, fitness ) );
    }

    return population;
}

// Function to perform genetic algorithm
template<typename Type>
void GeneticAlgorithm(const Type& target, const size_t& populationSize, const size_t& generations, const double& selectionPercentage, const size_t & maxDepth = 3, const double& prunningProbability = 0.0f) {
    srand( time( nullptr ) );

    // Initialize the population with random expression trees
    std::vector< std::pair<Expression<Type>*, double> > population = GeneratePopulation(target, populationSize, maxDepth, 0.0f);

    size_t lastGeneration = 0;
    double lastBest = 0.0f;

    // Main loop for generations
    for (size_t generation = 0; generation < generations; ++generation) {
        // Select the top individuals
        std::vector<Expression<Type>*> selectedPopulation;
        SelectTopIndividuals(population, selectedPopulation, selectionPercentage);

        // Reproduce the selected individuals
        std::vector<std::pair<Expression<Type>*, double>> newPopulation;
        newPopulation.reserve(populationSize);

        while (newPopulation.size() < populationSize) {
            // Randomly choose parents from the selected population
            size_t parentIndex1 = rand() % selectedPopulation.size();
            size_t parentIndex2 = rand() % selectedPopulation.size();

            Expression<Type>* child;

            // Perform crossover or mutation based on probability
            if (rand() % 2 == 0) {
                // Crossover
                child = Crossover(selectedPopulation[parentIndex1], selectedPopulation[parentIndex2]);
            } else {
                // Mutation
                child = Mutate(selectedPopulation[parentIndex1], maxDepth>>1, prunningProbability);
            }

            double fitness = EvaluateFitness(child, target);
            newPopulation.emplace_back( std::make_pair( child, fitness ));
        }

        // Clean up old population
        for (auto& entry : population) {
            delete entry.first;
        }

        // Update the population with the new one
        population = std::move(newPopulation);

        // Output the best individual in the current generation
        std::sort(population.begin(), population.end(),
                  [](const auto& a, const auto& b) {
                      return a.second > b.second;
                  });

        fprintf(stdout, "Generation %d, Best Fitness : %.6lf\n", generation+1, population.front().second);

        if(population.front().second >= 1.0f)
            break;
    }

    Expression<Type> * bestExpression = population.front().first;

    bestExpression->Show();
    fprintf(stdout, "\nValue : %lf\n", bestExpression->Evaluate());

    // Clean up the final population
    for (auto& entry : population) {
        delete entry.first;
    }
}


int main(int argc, char *argv[]){

    // Specify the target value and genetic algorithm parameters
    float targetValue = 42; //Now only float and double are handled properly
    size_t populationSize = 100;
    size_t generations = 1000;
    double selectionPercentage = 0.2;
    size_t maxDepth = 2;
    double prunningProbability = 0.1f;

    // Run the genetic algorithm
    GeneticAlgorithm(targetValue, populationSize, generations, selectionPercentage, maxDepth, prunningProbability);


    return 0;
}