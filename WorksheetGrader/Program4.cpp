#include "EndToEndWrapper.h"

int main(int argc, char* argv[])
{
	EndToEndWrapper e2e = EndToEndWrapper();
	char* nothing[1];
	e2e.run(0,nothing);
}