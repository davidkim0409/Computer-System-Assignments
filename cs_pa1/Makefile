CC ?= cc
# -ffp-contract=off keeps the float results reproducible across machines, so
# the example outputs can be compared exactly.
CFLAGS ?= -std=c11 -O2 -ffp-contract=off -Wall -Wextra -Wpedantic -Werror
CPPFLAGS = -Iinclude
LDLIBS = -lm
STUDENT = src/quant.c
HEADERS = include/quant.h

# Submission tarball. The login name on the assignment server is the student
# number with a leading 'u' (for example u2026314239), so strip that prefix.
# Building somewhere else? Pass it in:  make tar STUDENT_ID=2026314239
STUDENT_ID ?= $(patsubst u%,%,$(shell id -un))
TARBALL = u$(STUDENT_ID).tar

.PHONY: all test tar clean

all: main mnist

main: $(STUDENT) provided/main.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(STUDENT) provided/main.c -o $@ $(LDLIBS)

mnist: $(STUDENT) provided/mnist.c $(HEADERS) data/mnist_gemm.bin
	$(CC) $(CPPFLAGS) $(CFLAGS) $(STUDENT) provided/mnist.c -o $@ $(LDLIBS)

# Run both examples and compare with the expected output.
test: main mnist
	@fail=0; \
	for i in 1 2; do \
		if ./main < examples/input$$i.txt 2>&1 | \
		   diff -u examples/output$$i.txt - > /dev/null; then \
			echo "[PASS] examples/input$$i.txt"; \
		else \
			echo "[FAIL] examples/input$$i.txt"; \
			./main < examples/input$$i.txt 2>&1 | \
			  diff -u examples/output$$i.txt - | head -30; \
			fail=1; \
		fi; \
	done; \
	if [ $$fail -eq 0 ]; then echo ""; echo "both examples match"; \
		echo ""; echo "MNIST GEMM test:"; ./mnist; \
	else echo ""; echo "expected output is in examples/output*.txt"; exit 1; fi

tar: $(STUDENT)
	@case '$(STUDENT_ID)' in \
	  ''|*[!0-9]*) echo "STUDENT_ID='$(STUDENT_ID)' is not a student number."; \
	               echo "Run: make tar STUDENT_ID=<your student id>"; exit 1;; \
	esac
	@rm -f $(TARBALL)
	tar cvf $(TARBALL) $(STUDENT)
	@echo ""
	@echo "Created $(TARBALL). Upload this file to LearnUs."

clean:
	rm -f main mnist u*.tar *.o
