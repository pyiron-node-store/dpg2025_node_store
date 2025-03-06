from abc import ABC, abstractmethod

from ase.calculators.calculator import Calculator

class AseCalculatorConfig(ABC):
    @abstractmethod
    def get_calculator(self) -> Calculator:
        pass
