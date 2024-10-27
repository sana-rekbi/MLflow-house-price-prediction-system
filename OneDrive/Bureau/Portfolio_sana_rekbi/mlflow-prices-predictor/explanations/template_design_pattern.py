from abc import ABC, abstractmethod


# Step 1: Create an abstract base class
class DiningExperience(ABC):

    # The template method defining the skeleton of the dining experience
    def serve_dinner(self):
        self.serve_appetizer()
        self.serve_main_course()
        self.serve_dessert()
        self.serve_beverage()

    # Abstract methods to serve each course (to be implemented by subclasses)
    @abstractmethod
    def serve_appetizer(self):
        pass

    @abstractmethod
    def serve_main_course(self):
        pass

    @abstractmethod
    def serve_dessert(self):
        pass

    @abstractmethod
    def serve_beverage(self):
        pass


# Step 2: Create concrete classes that implement the template steps
class ItalianDinner(DiningExperience):
    def serve_appetizer(self):
        print("Serving bruschetta as appetizer.")

    def serve_main_course(self):
        print("Serving pasta as the main course.")

    def serve_dessert(self):
        print("Serving tiramisu as dessert.")

    def serve_beverage(self):
        print("Serving wine as the beverage.")


class ChineseDinner(DiningExperience):
    def serve_appetizer(self):
        print("Serving spring rolls as appetizer.")

    def serve_main_course(self):
        print("Serving stir-fried noodles as the main course.")

    def serve_dessert(self):
        print("Serving fortune cookies as dessert.")

    def serve_beverage(self):
        print("Serving tea as the beverage.")


# Step 3: Client code
if __name__ == "__main__":
    print("Italian Dinner:")
    italian_dinner = ItalianDinner()
    italian_dinner.serve_dinner()

    print("\nChinese Dinner:")
    chinese_dinner = ChineseDinner()
    chinese_dinner.serve_dinner()
