L = linspace(1, 3, 10);
V = L.^3
for i = 1:length(L)
  fprintf("Side length: %g, Volume: %g\n", L(i), V(i));
end
plot(L, V);
xlabel("Length (cm)")
ylabel("Volume (cm3)")
title("Cube volumes")
axis tight