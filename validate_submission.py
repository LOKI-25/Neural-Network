#!/usr/bin/env python3
"""
CS5720 Assignment 1: Neural Network Fundamentals - Submission Validator

This script validates student submissions for Assignment 1 to ensure they meet
all requirements before final submission.

Usage: python validate_submission.py your_student_id_pa1.py
Example: python validate_submission.py 700762718_pa1.py
"""

import os
import sys
import re
import ast
import importlib.util
import traceback
import time
from typing import List, Dict, Any, Optional


class CS5720Assignment1Validator:
    """Validator for CS5720 Assignment 1: Neural Network Fundamentals."""
    
    def __init__(self, filename: str):
        """Initialize validator with submission filename."""
        self.filename = filename
        self.basename = os.path.basename(filename)
        
        # Validation results
        self.errors = []
        self.warnings = []
        self.successes = []
        
        # Assignment 1 specific requirements
        self.required_classes = [
            'Layer', 'Dense', 'ReLU', 'Sigmoid', 'Softmax',
            'Loss', 'MSELoss', 'CrossEntropyLoss',
            'Optimizer', 'SGD', 'Momentum', 'NeuralNetwork'
        ]
        self.required_functions = [
            'load_mnist', 'one_hot_encode', 'gradient_check'
        ]
        
        # Assignment details
        self.assignment_name = "Neural Network Fundamentals"
        self.assignment_num = 1
    
    def validate_submission(self) -> bool:
        """Run complete validation check."""
        print(f"🔍 CS5720 Assignment 1 Submission Validator")
        print(f"📚 {self.assignment_name}")
        print(f"📁 Validating: {self.basename}")
        print("-" * 60)
        
        # Run validation checks
        self._validate_file_existence()
        self._validate_filename_format()
        self._validate_file_syntax()
        self._validate_required_classes()
        self._validate_required_functions()
        self._validate_imports()
        self._validate_student_info()
        self._check_code_quality()
        self._final_submission_check()
        
        # Generate report
        return self._generate_validation_report()
    
    def _validate_file_existence(self):
        """Check if file exists and is readable."""
        if not os.path.exists(self.filename):
            self.errors.append(f"❌ File not found: {self.filename}")
            return
        
        if not os.path.isfile(self.filename):
            self.errors.append(f"❌ Path is not a file: {self.filename}")
            return
            
        try:
            file_size = os.path.getsize(self.filename)
            if file_size == 0:
                self.errors.append("❌ File is empty")
                return
            elif file_size < 2000:
                self.warnings.append("⚠️ File seems very small - ensure all neural network components are implemented")
            elif file_size > 200000:
                self.warnings.append("⚠️ File is quite large - consider code optimization")
                
            self.successes.append("✅ File exists and has content")
            
        except Exception as e:
            self.errors.append(f"❌ Cannot access file: {str(e)}")
    
    def _validate_filename_format(self):
        """Validate filename follows required format."""
        pattern = r'^(\d{9})_pa1\.py$'
        match = re.match(pattern, self.basename)
        
        if not match:
            self.errors.append(
                f"❌ CRITICAL: Invalid filename format '{self.basename}'\n"
                f"   Expected format: {{9-digit-student-id}}_pa1.py\n"
                f"   Example: 700762718_pa1.py"
            )
            return
        
        student_id = match.group(1)
        
        # Validate student ID (9 digits)
        if len(student_id) != 9 or not student_id.isdigit():
            self.errors.append(f"❌ Invalid student ID format: {student_id} (must be exactly 9 digits)")
            return
        
        self.successes.append(f"✅ Filename format correct: {self.basename}")
        self.student_id = student_id
    
    def _validate_file_syntax(self):
        """Check for Python syntax errors."""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common encoding issues
            if '\ufeff' in content:  # BOM
                self.warnings.append("⚠️ File contains BOM (Byte Order Mark) - may cause issues")
            
            # Parse syntax
            ast.parse(content, filename=self.filename)
            self.successes.append("✅ Python syntax is valid")
            
        except SyntaxError as e:
            self.errors.append(
                f"❌ CRITICAL: Syntax error at line {e.lineno}: {e.msg}\n"
                f"   Fix this error before submission!"
            )
        except UnicodeDecodeError:
            self.errors.append("❌ CRITICAL: File encoding error - ensure file is saved as UTF-8")
        except Exception as e:
            self.errors.append(f"❌ Error reading file: {str(e)}")
    
    def _validate_required_classes(self):
        """Check for required neural network classes."""
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
            
            found_classes = []
            missing_classes = []
            
            for class_name in self.required_classes:
                class_pattern = rf'class\s+{class_name}\s*(?:\([^)]*\))?\s*:'
                if re.search(class_pattern, content):
                    found_classes.append(class_name)
                else:
                    missing_classes.append(class_name)
            
            if found_classes:
                self.successes.append(f"✅ Found classes: {', '.join(found_classes)}")
            
            if missing_classes:
                self.errors.append(
                    f"❌ CRITICAL: Missing required classes: {', '.join(missing_classes)}\n"
                    f"   Required classes: {', '.join(self.required_classes)}\n"
                    f"   Implement all classes with exact names (case-sensitive)"
                )
            else:
                self.successes.append("✅ All required classes found")
                
        except Exception as e:
            self.errors.append(f"❌ Error checking for required classes: {str(e)}")
    
    def _validate_required_functions(self):
        """Check for required utility functions."""
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
            
            found_functions = []
            missing_functions = []
            
            for func_name in self.required_functions:
                func_pattern = rf'def\s+{func_name}\s*\('
                if re.search(func_pattern, content):
                    found_functions.append(func_name)
                else:
                    missing_functions.append(func_name)
            
            if found_functions:
                self.successes.append(f"✅ Found functions: {', '.join(found_functions)}")
            
            if missing_functions:
                self.errors.append(
                    f"❌ CRITICAL: Missing required functions: {', '.join(missing_functions)}\n"
                    f"   Required functions: {', '.join(self.required_functions)}\n"
                    f"   Implement all functions with exact names (case-sensitive)"
                )
            else:
                self.successes.append("✅ All required functions found")
                
        except Exception as e:
            self.errors.append(f"❌ Error checking for required functions: {str(e)}")
    
    def _validate_imports(self):
        """Check for proper imports and prohibited libraries."""
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
            
            # Check for prohibited imports (libraries that would trivialize neural networks)
            prohibited_patterns = [
                r'from\s+(?:tensorflow|keras|torch|pytorch)',
                r'import\s+(?:tensorflow|keras|torch|pytorch)',
                r'from\s+sklearn\s+import.*neural_network',
                r'import\s+.*neural.*network.*library',
            ]
            
            prohibited_found = []
            for pattern in prohibited_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    prohibited_found.append(pattern)
            
            if prohibited_found:
                self.warnings.append(
                    f"⚠️ Potentially prohibited deep learning libraries detected\n"
                    f"   Assignment requires implementing neural networks from scratch\n"
                    f"   Avoid using TensorFlow, PyTorch, Keras, or sklearn neural networks"
                )
            
            # Check for valid imports
            valid_imports = ['import numpy', 'import math', 'import time', 'import random',
                           'from typing import', 'import copy', 'import pickle',
                           'import matplotlib', 'import gzip']
            
            found_valid = []
            for imp in valid_imports:
                if imp in content:
                    found_valid.append(imp.split()[-1])
            
            if found_valid:
                self.successes.append(f"✅ Standard imports found: {', '.join(found_valid)}")
            
        except Exception as e:
            self.warnings.append(f"⚠️ Could not validate imports: {str(e)}")
    
    def _validate_student_info(self):
        """Check for student identification in comments."""
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
            
            # Look for student ID in comments
            student_id_patterns = [
                r'Student ID.*?(\d{9})',
                r'ID.*?(\d{9})', 
                r'(\d{9})',  # Any 9-digit number
            ]
            
            found_id = False
            for pattern in student_id_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    found_id = True
                    break
            
            if found_id:
                self.successes.append("✅ Student ID found in file")
            else:
                self.warnings.append(
                    "⚠️ Student ID not clearly identified in comments\n"
                    "   Add a comment with your 9-digit student ID for identification"
                )
            
            # Look for student name
            name_patterns = [
                r'Student Name.*?:.*?([A-Z][a-z]+ [A-Z][a-z]+)',
                r'Name.*?:.*?([A-Z][a-z]+ [A-Z][a-z]+)',
                r'Author.*?:.*?([A-Z][a-z]+ [A-Z][a-z]+)'
            ]
            
            found_name = False
            for pattern in name_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found_name = True
                    break
            
            if found_name:
                self.successes.append("✅ Student name found in file")
            else:
                self.warnings.append("⚠️ Student name not found in comments")
                
        except Exception as e:
            self.warnings.append(f"⚠️ Could not validate student info: {str(e)}")
    
    def _check_code_quality(self):
        """Basic code quality checks."""
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for docstrings
            docstring_count = len(re.findall(r'""".*?"""', content, re.DOTALL))
            if docstring_count >= 6:
                self.successes.append("✅ Excellent documentation (multiple docstrings)")
            elif docstring_count >= 3:
                self.successes.append("✅ Good documentation found")
            else:
                self.warnings.append("⚠️ Consider adding docstrings for better documentation")
            
            # Check for comments
            comment_lines = [line for line in lines if line.strip().startswith('#')]
            if len(comment_lines) >= 20:
                self.successes.append("✅ Well-commented code")
            elif len(comment_lines) >= 10:
                self.successes.append("✅ Some comments found")
            else:
                self.warnings.append("⚠️ Consider adding more explanatory comments")
            
            # Check for neural network terminology
            nn_terms = ['forward', 'backward', 'gradient', 'weights', 'bias', 'activation', 'loss', 'optimizer']
            found_terms = [term for term in nn_terms if term.lower() in content.lower()]
            if found_terms:
                self.successes.append(f"✅ Neural network terminology found: {', '.join(found_terms)}")
            
            # Check for TODO/FIXME/PLACEHOLDER
            todo_patterns = [r'TODO', r'FIXME', r'PLACEHOLDER', r'#.*Remove.*this']
            todos_found = []
            for pattern in todo_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    todos_found.append(pattern)
            
            if todos_found:
                self.warnings.append(
                    f"⚠️ Development markers found: {', '.join(todos_found)}\n"
                    f"   Complete implementation before submission"
                )
                
        except Exception as e:
            self.warnings.append(f"⚠️ Could not perform code quality check: {str(e)}")
    
    def _final_submission_check(self):
        """Final submission readiness checks."""
        try:
            with open(self.filename, 'r') as f:
                content = f.read()
            
            # Look for starter code indicators
            starter_indicators = [
                'TODO:',
                'PLACEHOLDER',
                'NotImplementedError',
                'pass  # TODO',
                'raise NotImplementedError',
                'Replace with your implementation'
            ]
            
            found_indicators = []
            for indicator in starter_indicators:
                if indicator in content:
                    found_indicators.append(indicator)
            
            if found_indicators:
                self.warnings.append(
                    f"⚠️ Starter code elements detected: {', '.join(found_indicators)}\n"
                    f"   Ensure all methods are fully implemented"
                )
            
            # Check for essential neural network patterns
            nn_patterns = ['forward', 'backward', 'gradient', 'weights', 'bias', 'loss']
            found_patterns = []
            for pattern in nn_patterns:
                if pattern in content.lower():
                    found_patterns.append(pattern)
            
            if found_patterns:
                self.successes.append(f"✅ Neural network patterns found: {', '.join(found_patterns)}")
            else:
                self.warnings.append("⚠️ Consider implementing proper forward/backward propagation")
            
            # Check for numpy operations (expected for neural networks)
            numpy_patterns = ['np.', 'numpy.', 'dot(', 'matmul', 'reshape', 'transpose']
            found_numpy = []
            for pattern in numpy_patterns:
                if pattern in content:
                    found_numpy.append(pattern)
                    
            if found_numpy:
                self.successes.append(f"✅ NumPy operations detected: {', '.join(found_numpy)}")
                
        except Exception as e:
            self.warnings.append(f"⚠️ Could not perform final submission check: {str(e)}")
    
    def _generate_validation_report(self) -> bool:
        """Generate final validation report."""
        print("\n" + "="*60)
        print("📋 VALIDATION REPORT")
        print("="*60)
        
        # Count results
        total_checks = len(self.successes) + len(self.warnings) + len(self.errors)
        success_count = len(self.successes)
        
        # Show successes
        if self.successes:
            print(f"\n✅ PASSED CHECKS ({len(self.successes)}):")
            for success in self.successes:
                print(f"   {success}")
        
        # Show warnings
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")
        
        # Show errors
        if self.errors:
            print(f"\n❌ CRITICAL ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   {error}")
        
        # Final status
        print(f"\n" + "="*60)
        
        if self.errors:
            print("🚨 SUBMISSION STATUS: NOT READY")
            print("   ❌ CRITICAL ERRORS MUST BE FIXED BEFORE SUBMISSION")
            print("   📝 Fix all errors listed above")
            print("   🔄 Re-run validator after fixes")
            success_rate = 0
        else:
            success_rate = (success_count / max(1, total_checks)) * 100
            
            if success_rate >= 90:
                print("🎉 SUBMISSION STATUS: EXCELLENT!")
                print("   ✅ Your submission looks ready for grading")
            elif success_rate >= 75:
                print("✅ SUBMISSION STATUS: READY (with minor improvements)")
                print("   👍 Good to submit, consider addressing warnings")
            else:
                print("⚠️ SUBMISSION STATUS: ACCEPTABLE (needs improvement)")
                print("   📝 Address warnings to improve submission quality")
        
        # Statistics
        print(f"\n📊 VALIDATION SCORE: {success_count}/{total_checks} checks passed")
        if success_rate > 0:
            print(f"📈 SUCCESS RATE: {success_rate:.1f}%")
        
        # Next steps
        print(f"\n💡 NEXT STEPS:")
        if self.errors:
            print("   1. Fix all CRITICAL ERRORS listed above")
            print(f"   2. Re-run: python validate_submission.py {self.basename}")
            print("   3. Address warnings for better code quality")
            print("   4. Submit to Brightspace when ready!")
        else:
            print("   1. Review warnings and improve if possible")
            print(f"   2. Test thoroughly: python test_solution.py")
            print("   3. Submit to Brightspace Assignment Portal!")
        
        print(f"\n📚 ASSIGNMENT REQUIREMENTS:")
        print(f"   • Required classes: {', '.join(self.required_classes)}")
        print(f"   • Required functions: {', '.join(self.required_functions)}")
        print(f"   • Implement neural network components from scratch")
        print(f"   • Use NumPy for matrix operations")
        print(f"   • No external deep learning libraries allowed")
        
        print("="*60)
        
        return len(self.errors) == 0


def main():
    """Main function to run the validator."""
    if len(sys.argv) != 2:
        print("CS5720 Assignment 1 Submission Validator")
        print("Usage: python validate_submission.py <your_student_id_pa1.py>")
        print("Example: python validate_submission.py 700762718_pa1.py")
        sys.exit(1)
    
    filename = sys.argv[1]
    validator = CS5720Assignment1Validator(filename)
    
    is_valid = validator.validate_submission()
    
    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()