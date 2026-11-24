"""Expand every prompt with WordNet and generous borderline answers, then re-tier answers by how often
they appear in real text (wordfreq: Wikipedia, news, books, subtitles, web)."""
import re
import sys
import unicodedata

from nltk.corpus import wordnet as wn
from wordfreq import top_n_list, zipf_frequency

PATH = sys.argv[1]
VERBOSE = "-v" in sys.argv

def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("&", "and").replace("+", "plus").replace("#", "sharp")
    s = re.sub(r"^(the|a|an) ", "", s)
    return re.sub(r"[^a-z0-9]", "", s)

def freq(label):
    return zipf_frequency(label.lower(), "en")

# ---------- WordNet sources per prompt ----------
# ("hypo", synset): every kind of X.  ("inst", synset): named examples of X.  ("parts", synset): parts of X.
WN = {
 "Name a dog breed": [("hypo", "dog.n.01")],
 "Name a breed of cat": [("hypo", "domestic_cat.n.01")],
 "Name a breed of horse": [("hypo", "horse.n.01")],
 "Name a flower": [("hypo", "flower.n.01")],
 "Name a type of tree": [("hypo", "tree.n.01")],
 "Name a fruit": [("hypo", "edible_fruit.n.01"), ("hypo", "berry.n.01")],
 "Name a vegetable": [("hypo", "vegetable.n.01")],
 "Name a green vegetable": [("hypo", "greens.n.01"), ("hypo", "cabbage.n.01")],
 "Name a vegetable that grows underground": [("hypo", "root_vegetable.n.01")],
 "Name a bird": [("hypo", "bird.n.01")],
 "Name a bird that can't fly": [("hypo", "ratite.n.01"), ("hypo", "penguin.n.01")],
 "Name an insect": [("hypo", "insect.n.01")],
 "Name an animal that is a reptile": [("hypo", "reptile.n.01")],
 "Name a mammal that lives in the ocean": [("hypo", "cetacean.n.01"), ("hypo", "pinniped_mammal.n.01"), ("hypo", "sea_cow.n.01")],
 "Name a type of shark": [("hypo", "shark.n.01")],
 "Name a fish people eat": [("hypo", "food_fish.n.01"), ("hypo", "fish.n.02")],
 "Name a sea creature with no backbone": [("hypo", "mollusk.n.01"), ("hypo", "crustacean.n.01"), ("hypo", "coelenterate.n.01"), ("hypo", "echinoderm.n.01"), ("hypo", "sponge.n.04"), ("hypo", "polychaete.n.01"), ("hypo", "ctenophore.n.01")],
 "Name a dinosaur": [("hypo", "dinosaur.n.01")],
 "Name a wild cat": [("hypo", "big_cat.n.01"), ("hypo", "wildcat.n.03")],
 "Name a farm animal": [("hypo", "livestock.n.01"), ("hypo", "domestic_fowl.n.01"), ("hypo", "cattle.n.01"), ("hypo", "sheep.n.01"), ("hypo", "goat.n.01"), ("hypo", "swine.n.01")],
 "Name a word for a baby animal": [("hypo", "young_mammal.n.01"), ("hypo", "young_bird.n.01")],
 "Name a type of cheese": [("hypo", "cheese.n.01")],
 "Name a type of bread": [("hypo", "bread.n.01"), ("hypo", "quick_bread.n.01"), ("hypo", "bun.n.01")],
 "Name a type of pasta": [("hypo", "pasta.n.01"), ("hypo", "pasta.n.02")],
 "Name a dessert": [("hypo", "dessert.n.01"), ("hypo", "cake.n.03"), ("hypo", "cookie.n.01"), ("hypo", "pie.n.01"), ("hypo", "pastry.n.02")],
 "Name a candy": [("hypo", "candy.n.01")],
 "Name a breakfast cereal": [("hypo", "cereal.n.03")],
 "Name a breakfast food": [("hypo", "cereal.n.03")],
 "Name a type of sandwich": [("hypo", "sandwich.n.01")],
 "Name a kitchen utensil": [("hypo", "kitchen_utensil.n.01"), ("hypo", "cooking_utensil.n.01"), ("hypo", "cutlery.n.02")],
 "Name something you'd find in a kitchen": [("hypo", "kitchen_utensil.n.01"), ("hypo", "cooking_utensil.n.01"), ("hypo", "kitchen_appliance.n.01"), ("hypo", "home_appliance.n.01"), ("hypo", "cutlery.n.02"), ("hypo", "tableware.n.01"), ("hypo", "crockery.n.01"), ("hypo", "cookware.n.01"), ("hypo", "condiment.n.01")],
 "Name a musical instrument": [("hypo", "musical_instrument.n.01")],
 "Name an orchestra instrument": [("hypo", "stringed_instrument.n.01"), ("hypo", "wind_instrument.n.01"), ("hypo", "percussion_instrument.n.01")],
 "Name a sport": [("hypo", "sport.n.01")],
 "Name a sport played with a ball": [("hypo", "field_game.n.01"), ("hypo", "court_game.n.01")],
 "Name a type of dance": [("hypo", "dancing.n.01")],
 "Name a board game": [("hypo", "board_game.n.01")],
 "Name an item of clothing": [("hypo", "garment.n.01"), ("hypo", "headdress.n.01"), ("hypo", "footwear.n.02"), ("hypo", "glove.n.02")],
 "Name a type of shoe": [("hypo", "shoe.n.01"), ("hypo", "boot.n.01"), ("hypo", "footwear.n.02")],
 "Name a type of boat": [("hypo", "boat.n.01"), ("hypo", "ship.n.01"), ("hypo", "vessel.n.02")],
 "Name a vehicle with wheels": [("hypo", "wheeled_vehicle.n.01")],
 "Name a part of a car": [("parts", "car.n.01"), ("parts", "motor_vehicle.n.01"), ("parts", "automobile_engine.n.01")],
 "Name a part of the body": [("hypo", "external_body_part.n.01"), ("parts", "human_body.n.01")],
 "Name a bone in the human body": [("hypo", "bone.n.01")],
 "Name an organ in the human body": [("hypo", "internal_organ.n.01"), ("hypo", "gland.n.01")],
 "Name a part of a cell": [("hypo", "organelle.n.01")],
 "Name a part of a plant or flower": [("hypo", "plant_part.n.01"), ("hypo", "reproductive_structure.n.01")],
 "Name an emotion": [("hypo", "emotion.n.01")],
 "Name a color": [("hypo", "chromatic_color.n.01"), ("hypo", "achromatic_color.n.01")],
 "Name a shade of blue": [("hypo", "blue.n.01")],
 "Name a gemstone or mineral": [("hypo", "gem.n.02"), ("hypo", "mineral.n.01")],
 "Name a type of rock": [("hypo", "rock.n.02")],
 "Name a type of cloud": [("hypo", "cloud.n.02")],
 "Name a type of weather": [("hypo", "atmospheric_phenomenon.n.01"), ("hypo", "precipitation.n.03")],
 "Name a currency": [("hypo", "monetary_unit.n.01")],
 "Name a 3D shape": [("hypo", "solid.n.03"), ("hypo", "polyhedron.n.01")],
 "Name a shape with straight sides": [("hypo", "polygon.n.01")],
 "Name a programming language": [("hypo", "programming_language.n.01")],
 "Name a music genre": [("hypo", "popular_music.n.01"), ("hypo", "music_genre.n.01")],
 "Name a hobby": [("hypo", "hobby.n.01"), ("hypo", "pastime.n.01")],
 "Name a subject taught in school": [("hypo", "discipline.n.01")],
 "Name a holiday": [("hypo", "holiday.n.01")],
 "Name a crop farmers grow": [("hypo", "cereal.n.01"), ("hypo", "legume.n.01")],
 "Name a metal on the periodic table": [("hypo", "metallic_element.n.01")],
 "Name a word that means “big”": [("similar", "large.a.01"), ("similar", "big.s.01")],
 "Name something found in space": [("hypo", "celestial_body.n.01")],
 "Name a mythical creature": [("hypo", "mythical_monster.n.01"), ("hypo", "imaginary_being.n.01")],
 "Name an author": [("inst", "writer.n.01")],
 "Name a scientist": [("inst", "scientist.n.01")],
 "Name a painter": [("inst", "painter.n.01")],
 "Name a composer": [("inst", "composer.n.01")],
 "Name an explorer": [("inst", "explorer.n.01")],
 "Name an athlete": [("inst", "athlete.n.01")],
 "Name a river": [("inst", "river.n.01")],
 "Name a lake": [("inst", "lake.n.01")],
 "Name a mountain": [("inst", "mountain.n.01"), ("inst", "volcano.n.02")],
 "Name a volcano": [("inst", "volcano.n.02")],
 "Name a sea": [("inst", "sea.n.01")],
 "Name a constellation": [("inst", "constellation.n.02")],
 "Name a Greek god or goddess": [("inst", "greek_deity.n.01")],
 "Name a Roman god or goddess": [("inst", "roman_deity.n.01")],
 "Name a Norse god or goddess": [("inst", "norse_deity.n.01")],
 "Name an ancient Egyptian god or goddess": [("inst", "egyptian_deity.n.01")],
 "Name a famous bridge": [("inst", "bridge.n.01")],
 "Name a US city": [("uscity", "city.n.01"), ("uscity", "town.n.01")],
}

# ---------- Generous borderline answers people actually type ----------
LENIENT = {
 "Name something you'd find in a kitchen": "Cookie cutter/Cookie cutters;Baking sheet/Cookie sheet;Cake pan;Muffin tin;Pie dish;Casserole dish;Cast iron skillet;Stand mixer;Hand mixer;Electric kettle;Coffee grinder;Espresso machine;French press;Bread maker;Popcorn maker;Ice maker;Ice cube tray;Water filter;Napkins;Placemats;Tablecloth;Salt shaker;Pepper grinder;Sugar bowl;Butter dish;Cookie jar;Bread box;Fruit basket;Paper plates;Plastic cups;Straws;Cans;Cereal box;Milk;Eggs;Butter;Cheese;Bananas;Apples;Leftovers;Snacks;Chips;Cookies;Honey;Ketchup;Mustard;Mayo/Mayonnaise;Olive oil;Vinegar;Soy sauce;Hot sauce;Recipe;Recipe box;Burner;Gas stove;Grill pan;Griddle;Steamer;Roasting pan;Baking dish;Ramekin;Measuring spoons;Kitchen scale;Meat thermometer;Pot holder;Trivet;Dishwasher pods;Scrub brush;Steel wool;Rubber gloves;Mop;Broom;Dustpan;Trash bags;Recycling bin;Compost bin;Bottle opener;Jar opener;Zester;Salad bowl;Serving spoon;Carving knife;Knife sharpener;Pizza cutter;Pizza stone;Parchment paper;Wax paper;Cheesecloth;Baster;Funnel;Sieve;Strainer;Splatter guard;Lid/Lids/Pot lid;Egg timer;Toothpicks;Skewers;Cooling rack;Cake stand;Piping bag;Sprinkles;Food coloring;Flour sifter/Sifter;Pastry cutter;Cookbook;Chef;Cook;Mom;Dad;Family;Dog;Cat;Ants;Mouse;Fruit flies;Plants;Herbs;Curtains;Chairs;Bar stools;High chair;Water bottle;Thermos;Sippy cup;Baby bottle;Bib;Bulletin board;Phone charger;Radio;TV;Speaker;Light switch;Outlet;Ceiling fan;Smoke detector;Cupboard;Drawer;Shelf;Lazy Susan;Paper towel holder;Soap dispenser;Hand soap;Dish brush;Cutting boards;Knives;Forks;Spoons;Plates;Bowls;Cups;Glasses;Mugs;Pots;Pans;Utensils;Silverware;Dishes;Water;Ice;Coffee;Tea;Juice;Soda;Fruit;Vegetables;Meat;Rice;Pasta;Cereal;Oil;Spatulas;Ladles;Ovens;Stoves;Fridge magnets",
 "Name something you'd find in a classroom": "Glue stick;Construction paper;Dry erase marker;Whiteboard eraser;Chalk eraser;Index cards;Flash cards;Paper towels;First aid kit;Lost and found;Coat hooks;Backpacks;Bean bags;Reading nook;Library;Books;Dictionary;Thesaurus;Atlas;Encyclopedia;Novels;Comic books;Magazines;Anchor charts;Alphabet chart;Multiplication chart;Screen;Microphone;Charging cart;Copier;Planner;Agenda;Report cards;Red pen;Stamp;Stickers;Prize box/Treasure box;Timer;Whistle;Yardstick;Meter stick;Protractor;Compass;Graph paper;Staples;Push pins;Three-hole punch;Pencil cup;Pencil box;Mechanical pencil;Oil pastels;Paintbrush;Clay;Play-Doh;Legos;Blocks;Puzzles;Board games;Cards;Dice;Counters;Manipulatives;Abacus;Base ten blocks;Fraction tiles;Magnifying glass;Beaker;Test tubes;Goggles;Lab coat;Skeleton;Model;Solar system model;Fish tank/Aquarium;Guinea pig;Terrarium;Blinds;Lock;Fan;Heater;Air conditioner;Water fountain;Recycling bin;Broom;Teacher's desk;Locker;Coat;Jacket;Umbrella;Principal;Substitute teacher;Janitor;Pencils;Pens;Markers;Crayons;Erasers;Notebooks;Desks;Chairs;Computers;Laptops;Windows;Posters;Maps;Rulers;Scissors;Calculators;Papers;Worksheets;Folders;Binders;Shelves;Bookcase;Cubby",
 "Name something you find at the beach": "Sandals;Flip flops;Towels;Umbrellas;Chairs;Cooler;Sunscreen;Sunglasses;Hat;Buckets;Shovels;Sandcastles;Shells;Starfish;Crabs;Seagulls;Pelicans;Waves;Tide;Ocean;Surf;Surfboards;Kayak;Jet ski;Boats;Sailboat;Dock;Lifeguard stand;Flags;Lighthouse;Dunes;Sea oats;Grass;Palm trees;Coconuts;Seaweed;Kelp;Driftwood;Rocks;Pebbles;Tide pools;Jellyfish;Fish;Dolphins;Sharks;Seals;Sea lions;Turtles;Footprints;Sand dollars;Sea glass;Shark teeth;Volleyball;Frisbee;Kite;Beach ball;Radio;Speaker;Book;Magazine;Snacks;Sandwiches;Ice cream;Snow cones;Lemonade;Water;Hot dogs;Food truck;Boardwalk;Arcade;Ferris wheel;Souvenir shop;Swimsuit;Bikini;Swim trunks;Rash guard;Wetsuit;Goggles;Snorkel;Flippers;Mask;Life jacket;Pool noodle;Floaties;Inner tube;Raft;Tent;Cabana;Blanket;Sunburn;Tan;Sand;Sun;Water;Birds;People;Kids;Dogs;Shells;Clams;Mussels;Oysters;Barnacles;Plankton;Coral;Reef;Cliffs;Cove;Bay;Island;Horizon;Sunset;Sunrise;Clouds;Wind;Salt;Bonfire;Fishing rod;Fishermen;Surfers;Swimmers;Tourists;Lifeguards;Trash;Bottles;Message in a bottle;Treasure;Pirates",
 "Name something you see at Halloween": "Jack-o'-lantern;Pumpkins;Gourds;Corn stalks;Scarecrows;Candy;Lollipops;Caramel apples;Popcorn balls;Costumes;Makeup;Face paint;Wigs;Capes;Witch hats;Broomsticks;Cauldrons;Ghosts;Skeletons;Skulls;Bones;Zombies;Vampires;Werewolves;Mummies;Frankenstein;Monsters;Witches;Wizards;Goblins;Ghouls;Demons;Devils;Angels;Fairies;Princesses;Superheroes;Pirates;Ninjas;Cowboys;Clowns;Black cats;Bats;Spiders;Spider webs;Cobwebs;Owls;Crows;Ravens;Rats;Snakes;Worms;Coffins;Tombstones;Graveyards;Haunted houses;Fog;Full moon;Candles;Lanterns;String lights;Decorations;Inflatables;Trick-or-treaters;Candy bags;Buckets;Doorbells;Porches;Parties;Costume contest;Scary movies;Haunted hayride;Corn maze;Pumpkin patch;Apple cider;Donuts;Fall leaves;Hay;Werewolf;Ghost;Witch;Vampire;Zombie;Skeleton;Mummy;Bat;Spider;Cat;Moon;Stars;Night;Darkness;Screams;Kids;Parents;Neighbors",
 "Name something that is yellow": "Bananas;Lemons;Sunflowers;School buses;Taxis;Rubber ducks;Chicks;Canaries;Goldfinch;Bees;Pencils;Highlighters;Sticky notes;Mustard;Pineapple;Mango;Squash;Daffodils;Dandelions;Tulips;Roses;Marigolds;Buttercups;Lemonade;Popcorn;Mac and cheese;Tennis balls;Smiley face;Emoji;Caution sign;Yield sign;Traffic light;Hard hat;Raincoat;Rain boots;Bulldozer;Crayon;Gold;Sulfur;Star;Stars;Legal pad;Caution tape;Honey;Corn chips;French fries;Cheese;Butter;Egg yolk;Yolk;Corn;Sun;Lemon;Banana;Sunflower;Bus;Pencil;Highlighter;Duck;Chick;Canary;Bee;Yellow jacket;Banana peel;Grapefruit;Pears;Yellow pepper;Yellow squash;Pasta;Noodles;Cornbread;Pancakes;Waffles;Bread;Custard;Lemon cake;Lemon bar;Candy;Gummy bears;Lemon drops;Star fruit;Golden retriever;Lab;Lion;Giraffe;Cheetah;Leopard;Tiger;Frog;Snake;Fish;Butterfly;Moth;Caterpillar;Wasp;Hornet;Firefly;Leaves;Fall leaves;Hay;Straw;Wheat;Sand;Desert",
 "Name something that has wings": "Birds;Planes;Airplanes;Butterflies;Bats;Bees;Flies;Angels;Dragons;Dragonflies;Fairies;Moths;Eagles;Hawks;Owls;Crows;Parrots;Chickens;Ducks;Geese;Swans;Penguins;Ostriches;Emus;Flamingos;Pigeons;Doves;Robins;Bluejays;Cardinals;Hummingbirds;Seagulls;Pelicans;Bugs;Insects;Ladybugs;Wasps;Hornets;Mosquitoes;Gnats;Beetles;Moths;Fireflies;Cicadas;Grasshoppers;Crickets;Locusts;Cockroaches;Termites;Ants;Jets;Helicopter;Drone;Rocket;Glider;Hang glider;Kite;Pterodactyl;Pegasus;Griffin;Phoenix;Cupid;Gargoyle;Wing nut;Chicken wings;Buffalo wings;Hospital wing;Building;Airport;Red Bull;Stage",
 "Name a type of weather": "Rainy;Snowy;Sunny;Cloudy;Windy;Stormy;Foggy;Hot;Cold;Warm;Cool;Freezing;Humid;Dry;Wet;Icy;Clear;Partly cloudy;Mostly sunny;Overcast;Thunder;Lightning;Rain;Snow;Hail;Sleet;Fog;Mist;Drizzle;Downpour;Shower;Flurries;Storm;Thunderstorm;Hurricane;Tornado;Typhoon;Cyclone;Blizzard;Heat wave;Cold snap;Drought;Flood;Monsoon;Sandstorm;Dust storm;Rainbow;Frost;Dew",
 "Name a hobby": "Sleeping;Eating;Shopping;Walking;Jogging;Lifting;Working out;Gym;Exercise;Biking;Skating;Surfing;Climbing;Skiing;Snowboarding;Boxing;Wrestling;Golfing;Bowling;Billiards;Darts;Fishing;Hunting;Camping;Hiking;Gardening;Cooking;Baking;Reading;Writing;Drawing;Painting;Sketching;Sculpting;Pottery;Knitting;Sewing;Crocheting;Quilting;Woodworking;Collecting;Photography;Filming;Editing;Gaming;Coding;Streaming;Singing;Dancing;Acting;Playing piano;Playing guitar;Playing drums;Listening to music;Watching movies;Watching TV;Watching anime;Reading manga;Journaling;Blogging;Vlogging;Volunteering;Traveling;Puzzles;Crosswords;Sudoku;Chess;Cards;Poker;Magic;Juggling;Origami;Calligraphy;Lego;Model trains;Bird watching;Stargazing;Kite flying;Geocaching;Metal detecting;Rock collecting;Coin collecting;Stamp collecting;Card collecting;Sports;Basketball;Soccer;Football;Baseball;Tennis;Volleyball;Swimming;Running;Yoga;Martial arts;Karate;Archery;Horse riding;Fishkeeping;Beekeeping;Bonsai;Sleep;Shopping",
 "Name an emotion": "Happy;Sad;Mad;Angry;Scared;Afraid;Excited;Nervous;Anxious;Worried;Jealous;Envious;Proud;Ashamed;Guilty;Embarrassed;Lonely;Bored;Confused;Surprised;Shocked;Disgusted;Grateful;Thankful;Hopeful;Hopeless;Calm;Relaxed;Content;Satisfied;Frustrated;Annoyed;Irritated;Furious;Upset;Hurt;Heartbroken;Overwhelmed;Stressed;Tired;Exhausted;Sleepy;Hungry;Silly;Goofy;Playful;Loving;Caring;Kind;Brave;Confident;Shy;Insecure;Curious;Interested;Amazed;Joyful;Cheerful;Glad;Delighted;Thrilled;Ecstatic;Miserable;Depressed;Gloomy;Grumpy;Moody;Cranky;Fearful;Terrified;Panicked;Peaceful;Love;Hate;Joy;Fear;Anger;Sadness;Happiness;Excitement",
 "Name a subject taught in school": "Gym;PE;Recess;Lunch;Homeroom;Advisory;ELA;Language arts;Reading;Writing;Spelling;Handwriting;Phonics;Math;Mathematics;Maths;Algebra 1;Algebra 2;Geometry;Trigonometry;Precalculus;Calculus;Statistics;Science;Biology;Chemistry;Physics;Earth science;Life science;Physical science;Environmental science;Anatomy;Astronomy;Geology;Social studies;History;US history;World history;Geography;Civics;Government;Economics;Psychology;Sociology;Art;Music;Band;Choir;Orchestra;Drama;Theater;Dance;Health;Spanish;French;German;Chinese;Latin;Japanese;Italian;ASL;Sign language;Computer science;Coding;Technology;Engineering;Robotics;Shop;Woodshop;Home ec;Cooking;Business;Accounting;Marketing;Journalism;Yearbook;Debate;Speech;Creative writing;Literature;English;Study hall;Library;STEM",
 "Name a breakfast food": "Eggs;Bacon;Sausage;Toast;Pancakes;Waffles;French toast;Cereal;Oatmeal;Grits;Yogurt;Granola;Fruit;Bananas;Berries;Strawberries;Blueberries;Apples;Orange juice;Milk;Coffee;Tea;Hash browns;Home fries;Breakfast potatoes;Bagels;Muffins;Croissants;Biscuits;Gravy;Donuts;Pastries;Cinnamon rolls;Scones;English muffins;Breakfast burrito;Breakfast sandwich;Omelet;Frittata;Quiche;Smoothie;Protein shake;Peanut butter toast;Avocado toast;Ham;Steak;Fried rice;Leftovers;Pizza;Cold pizza;Pop-Tarts;Toaster waffles;Eggo;Cereal bar;Granola bar;Protein bar;Jam;Jelly;Syrup;Maple syrup;Butter;Cream cheese;Peanut butter;Nutella;Cheese;Sausage links;Turkey bacon;Chorizo;Huevos rancheros;Breakfast tacos;Crepes;Dutch baby;Eggs Benedict;Poached eggs;Boiled eggs;Scrambled eggs;Fried eggs;Sunny side up",
}

def lemma_label(name):
    s = name.replace("_", " ")
    return s[0].upper() + s[1:] if s and s[0].islower() else s

def good_lemma(name):
    words = name.split("_")
    if len(words) > 4 or len(name) > 40 or re.search(r"\d", name):
        return False
    # Skip Latin-looking names that never appear in everyday text.
    return all(zipf_frequency(w.lower(), "en") > 0 for w in words)

# Prompts where WordNet's capitalized entries are brand or variety names (Empire apple, Jonathan) rather than kinds.
COMMON_ONLY = {"Name a fruit", "Name a vegetable", "Name a green vegetable", "Name a vegetable that grows underground",
               "Name a flower", "Name a type of tree", "Name something you'd find in a kitchen", "Name a kitchen utensil",
               "Name a crop farmers grow", "Name an emotion", "Name a color", "Name a type of weather", "Name a hobby"}

def synset_groups(kind, sname, common_only=False, people=False):
    try:
        root = wn.synset(sname)
    except Exception:
        print("  missing synset", sname)
        return []
    groups = []
    if kind == "hypo":
        # Keep specific kinds (leaves) plus one-word groups like "terrier"; drop umbrella terms like "hunting dog".
        syns = {s for s in root.closure(lambda s: s.hyponyms())
                if not s.hyponyms() or "_" not in s.lemmas()[0].name()}
    elif kind == "inst":
        types = {root} | set(root.closure(lambda s: s.hyponyms()))
        syns = {i for t in types for i in t.instance_hyponyms()}
    elif kind == "parts":
        syns = set(root.closure(lambda s: s.part_meronyms())) | {h for p in root.part_meronyms() for h in p.hyponyms()}
    elif kind == "similar":
        syns = set(root.similar_tos()) | {root}
    elif kind == "uscity":
        us = wn.synset("united_states.n.01")
        types = {root} | set(root.closure(lambda s: s.hyponyms()))
        syns = {i for t in types for i in t.instance_hyponyms() if us in set(i.closure(lambda s: s.part_holonyms()))}
    for s in syns:
        names = [l.name() for l in s.lemmas() if good_lemma(l.name()) and not (common_only and l.name()[0].isupper())]
        if names:
            if kind == "inst" and people:
                # Full names first ("Thomas Young", not "Young"), so the label and its frequency mean the person.
                names.sort(key=lambda n: (-n.count("_"), -len(n)))
            else:
                names.sort(key=lambda n: -zipf_frequency(n.replace("_", " ").lower(), "en"))
            groups.append([lemma_label(n) for n in names])
    return groups

def parse(raw):
    out = []
    for part in raw.split(";"):
        names, tier = part.rsplit("*", 1)
        out.append({"names": names.split("/"), "prior": int(tier), "curated": True})
    return out

def add(answers, names, prior):
    taken = {}
    for a in answers:
        for n in a["names"]:
            taken[norm(n)] = a
    names = [n for n in names if norm(n) and not re.search(r'[;*/"\\]', n)]
    if not names:
        return
    hit = next((taken[norm(n)] for n in names if norm(n) in taken), None)
    if hit:
        for n in names:
            if norm(n) not in taken:
                hit["names"].append(n); taken[norm(n)] = hit
    else:
        fresh, seen = [], set()
        for n in names:
            if norm(n) not in seen:
                fresh.append(n); seen.add(norm(n))
        answers.append({"names": fresh, "prior": prior, "curated": False})

# Tier = how far an answer's real-world frequency falls below the most-mentioned answers for that prompt.
# Zipf is log10 scale, so a gap of 1.0 means the answer shows up about 10x less often.
CUTS = [(.5, 1), (1.1, 2), (2.1, 3), (2.9, 4)]

def retier(q, answers):
    for a in answers:
        label = a["names"][0]
        z = max(freq(label), 0.5)
        # wordfreq estimates a phrase from its words, which overstates phrases ("water dog" looks as common as "dog").
        z -= 0.5 * (len(label.split()) - 1)
        # Common words used in an unusual sense (Pan, Nike, Life) would otherwise look common, so lean a little on
        # the hand-made tier for curated answers, and nudge auto-added ones down.
        if a["curated"]:
            z -= 0.25 * (a["prior"] - 1)
        else:
            z -= 0.5 if a["prior"] == 5 else 0.25
        a["score"] = z
    ranked = sorted(answers, key=lambda a: -a["score"])
    ref = sum(a["score"] for a in ranked[:3]) / min(3, len(ranked))
    for a in ranked:
        gap = ref - a["score"]
        data_tier = next((t for cut, t in CUTS if gap < cut), 5)
        if a["curated"]:
            # Average the text-frequency tier with the hand-made one; on a tie, trust the data.
            a["tier"] = min(5, (data_tier + min(a["prior"], 5) + (1 if data_tier > a["prior"] else 0)) // 2)
        else:
            # Auto-added answers can't be "common" just because the word has another everyday meaning.
            floor = 4 if " " not in a["names"][0] and freq(a["names"][0]) >= 3.5 else 3
            a["tier"] = max(data_tier, floor)
    # Exactly one Trench answer: the least-mentioned hand-picked answer that still shows up in real text.
    curated = [a for a in ranked if a["curated"]]
    real = [a for a in curated if freq(a["names"][0]) >= 1.0] or curated
    real[-1]["tier"] = 6
    return ranked

html = open(PATH, encoding="utf-8").read()
start = html.index("const BANK = [\n") + len("const BANK = [\n")
end = html.index("\n];", start)
rows = []
ology = [w for w in top_n_list("en", 300000) if w.endswith("ology") and len(w) > 6 and w.isalpha()]
for line in html[start:end].split("\n"):
    m = re.match(r'^ \["(.*?)","(.*)"\],$', line)
    q, answers = m.group(1), parse(m.group(2))
    before = len(answers)
    for kind, sname in WN.get(q, []):
        people = bool(re.search(r"scientist|painter|composer|explorer|author|athlete", q))
        for names in synset_groups(kind, sname, q in COMMON_ONLY, people):
            add(answers, names, 5)
    for part in LENIENT.get(q, "").split(";"):
        if part:
            add(answers, part.split("/"), 4)
    if q == "Name a word that ends in -ology":
        for w in ology:
            add(answers, [w.capitalize()], 5)
    ranked = retier(q, answers)
    rows.append((q, ranked))
    if VERBOSE:
        counts = [sum(1 for a in ranked if a["tier"] == t) for t in range(1, 7)]
        print(f"{q[:48]:48} {before:4} -> {len(ranked):5}  tiers {counts}")

total = 0
out = []
for q, answers in rows:
    total += len(answers)
    assert sum(1 for a in answers if a["tier"] == 6) == 1, q
    keys = [k for a in answers for k in {norm(n) for n in a["names"]}]
    assert len(keys) == len(set(keys)), q
    out.append(f' ["{q}","' + ";".join("/".join(a["names"]) + "*" + str(a["tier"]) for a in answers) + '"],')
html = html[:start] + "\n".join(out) + html[end:]
open(PATH, "w", encoding="utf-8").write(html)
print(f"prompts {len(rows)}, answers {total}")
